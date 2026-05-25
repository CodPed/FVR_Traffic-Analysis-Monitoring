import os
import time
import threading
from collections import Counter

import gi
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds


VIDEO_PATH = "/data/traffic.mp4"
DISTANCIA_METROS = 16.0

# As bounding boxes do DeepStream estão no espaço do nvstreammux.
MUX_W = 1920
MUX_H = 1080

veiculos = {}
cv_cap = None
cv_lock = threading.Lock()
frame_cache = {"num": -1, "frame": None}


def init_cv2():
    global cv_cap

    if cv2 is None:
        print("[AVISO] OpenCV/cv2 não está disponível. A cor ficará como 'cor desconhecida'.")
        return

    cv_cap = cv2.VideoCapture(VIDEO_PATH)

    if not cv_cap.isOpened():
        print(f"[AVISO] Não consegui abrir {VIDEO_PATH} com OpenCV. A cor ficará como 'cor desconhecida'.")
        cv_cap = None
        return

    w = int(cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cv_cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] OpenCV ativo para estimativa de cor. Vídeo: {w}x{h} @ {fps:.2f} FPS")


def obter_frame_cv2(frame_num):
    global cv_cap, frame_cache

    if cv2 is None or cv_cap is None:
        return None

    try:
        with cv_lock:
            if frame_cache["num"] == frame_num and frame_cache["frame"] is not None:
                return frame_cache["frame"]

            cv_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num)))
            ok, frame = cv_cap.read()

            if not ok or frame is None:
                return None

            frame_cache["num"] = frame_num
            frame_cache["frame"] = frame
            return frame

    except Exception:
        return None


def obter_crop_objeto(frame_bgr, obj_meta):
    if frame_bgr is None:
        return None

    try:
        frame_h, frame_w = frame_bgr.shape[:2]
        sx = frame_w / float(MUX_W)
        sy = frame_h / float(MUX_H)

        r = obj_meta.rect_params
        x1 = int(max(0, r.left * sx))
        y1 = int(max(0, r.top * sy))
        x2 = int(min(frame_w, (r.left + r.width) * sx))
        y2 = int(min(frame_h, (r.top + r.height) * sy))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return None

        return frame_bgr[y1:y2, x1:x2]

    except Exception:
        return None


def obter_pixels_carrocaria(crop_bgr):
    """
    Tenta isolar a carroçaria antes de classificar a cor.
    Usa GrabCut quando possível e fallback para zona central.
    """
    if crop_bgr is None or crop_bgr.size == 0 or cv2 is None:
        return None

    h, w = crop_bgr.shape[:2]
    if w < 30 or h < 30:
        return None

    # Zona útil do veículo: evita estrada/pneus em baixo e margens.
    roi = crop_bgr[
        int(h * 0.08):int(h * 0.72),
        int(w * 0.10):int(w * 0.90)
    ].copy()

    if roi.size == 0:
        return None

    rh, rw = roi.shape[:2]

    try:
        # Reduz para acelerar o GrabCut.
        max_side = 180
        if max(rh, rw) > max_side:
            scale = max_side / float(max(rh, rw))
            small = cv2.resize(roi, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = roi

        sh, sw = small.shape[:2]
        if sw < 20 or sh < 20:
            return None

        mask = np.zeros((sh, sw), np.uint8)
        rect = (
            int(sw * 0.08),
            int(sh * 0.08),
            max(1, int(sw * 0.84)),
            max(1, int(sh * 0.84)),
        )
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)

        fg = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        if scale != 1.0:
            fg = cv2.resize(fg, (rw, rh), interpolation=cv2.INTER_NEAREST)

    except Exception:
        fg = np.zeros((rh, rw), np.uint8)
        fg[int(rh * 0.15):int(rh * 0.72), int(rw * 0.18):int(rw * 0.82)] = 255

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Remove vegetação e sombras fortes.
    relva = (H >= 35) & (H <= 85) & (S > 55) & (V > 45)
    sombras = V < 25

    final_mask = (fg > 0) & (~relva) & (~sombras)
    pixels = roi[final_mask]

    if pixels.shape[0] < 40:
        fallback = roi[
            int(rh * 0.15):int(rh * 0.68),
            int(rw * 0.20):int(rw * 0.80)
        ]
        if fallback.size == 0:
            return None
        pixels = fallback.reshape(-1, 3)

    return pixels


def classificar_cor(crop_bgr):
    """
    Classificação por percentagem de pixels da carroçaria.
    Classes finais: branco, preto, cinzento/prata, vermelho, azul, amarelo/laranja.
    """
    if crop_bgr is None or crop_bgr.size == 0 or cv2 is None:
        return None

    try:
        pixels = obter_pixels_carrocaria(crop_bgr)
        if pixels is None or len(pixels) < 40:
            return None

        pixels = pixels.reshape(-1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        H = hsv[:, 0]
        S = hsv[:, 1]
        V = hsv[:, 2]
        total = len(H)

        if total < 40:
            return None

        scores = {
            "branco": np.sum((V > 170) & (S < 80)) / total,
            "preto": np.sum(V < 55) / total,
            "cinzento/prata": np.sum((S < 60) & (V >= 55) & (V <= 205)) / total,
            "vermelho": np.sum(((H < 10) | (H > 170)) & (S > 85) & (V > 60)) / total,
            "azul": np.sum((H >= 90) & (H <= 130) & (S > 75) & (V > 50)) / total,
            "amarelo/laranja": np.sum((H >= 18) & (H <= 38) & (S > 75) & (V > 70)) / total,
        }

        cor, score = max(scores.items(), key=lambda x: x[1])

        # Thresholds separados para reduzir falsos positivos.
        if cor in ("branco", "preto", "cinzento/prata"):
            if score < 0.26:
                return None
        else:
            if score < 0.16:
                return None

        return cor

    except Exception:
        return None


def atualizar_cor_veiculo(v_data, cor):
    if cor is None:
        return

    v_data["cores_votos"].append(cor)

    if len(v_data["cores_votos"]) > 30:
        v_data["cores_votos"] = v_data["cores_votos"][-30:]

    contagem = Counter(v_data["cores_votos"])
    cor_mais_comum, votos = contagem.most_common(1)[0]

    if votos >= 3:
        v_data["cor"] = cor_mais_comum


def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        agora = time.time()
        frame_num = int(frame_meta.frame_num)
        frame_bgr = None

        l_obj = frame_meta.obj_meta_list

        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_id = obj_meta.object_id
            id_curto = obj_id & 0xFFFF

            if obj_id not in veiculos:
                veiculos[obj_id] = {
                    "t_topo": None,
                    "t_base": None,
                    "velocidade": None,
                    "contramao": False,
                    "lado": None,
                    "alertado": False,
                    "cor": None,
                    "cores_votos": [],
                    "tentativas_cor": 0,
                }

            v_data = veiculos[obj_id]

            if v_data["tentativas_cor"] < 35:
                if frame_bgr is None:
                    frame_bgr = obter_frame_cv2(frame_num)

                crop = obter_crop_objeto(frame_bgr, obj_meta)
                cor = classificar_cor(crop)

                v_data["tentativas_cor"] += 1
                atualizar_cor_veiculo(v_data, cor)

            l_user = obj_meta.obj_user_meta_list

            while l_user:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                    try:
                        analytics = pyds.NvDsAnalyticsObjMeta.cast(user_meta.user_meta_data)
                    except AttributeError:
                        analytics = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)

                    status = analytics.roiStatus

                    if any(x in status for x in ["L_Topo", "roi-L_Topo"]):
                        if v_data["t_topo"] is None:
                            v_data["t_topo"] = agora
                            v_data["lado"] = "L"

                    elif any(x in status for x in ["R_Topo", "roi-R_Topo"]):
                        if v_data["t_topo"] is None:
                            v_data["t_topo"] = agora
                            v_data["lado"] = "R"

                    if any(x in status for x in ["L_Base", "roi-L_Base"]):
                        if v_data["t_base"] is None:
                            v_data["t_base"] = agora
                            v_data["lado"] = "L"

                    elif any(x in status for x in ["R_Base", "roi-R_Base"]):
                        if v_data["t_base"] is None:
                            v_data["t_base"] = agora
                            v_data["lado"] = "R"

                    if v_data["velocidade"] is None and v_data["t_topo"] and v_data["t_base"]:
                        dt = abs(v_data["t_topo"] - v_data["t_base"])

                        if dt > 0.3:
                            v_data["velocidade"] = int((DISTANCIA_METROS / dt) * 3.6)

                            if (v_data["lado"] == "L" and v_data["t_base"] < v_data["t_topo"]) or \
                               (v_data["lado"] == "R" and v_data["t_topo"] < v_data["t_base"]):
                                v_data["contramao"] = True

                            cor_txt = v_data["cor"] or "cor desconhecida"

                            if v_data["contramao"] and not v_data["alertado"]:
                                print(
                                    f"\033[91m[ALERTA CRÍTICO]\033[0m "
                                    f"Veículo ID:{id_curto} ({cor_txt}) detetado em CONTRAMÃO "
                                    f"a {v_data['velocidade']} km/h!"
                                )
                                v_data["alertado"] = True

                            elif not v_data["contramao"] and not v_data["alertado"]:
                                print(
                                    f"[INFO] Veículo ID:{id_curto} ({cor_txt}) passou corretamente "
                                    f"a {v_data['velocidade']} km/h."
                                )
                                v_data["alertado"] = True

                l_user = l_user.next

            cor_txt = v_data["cor"] or "cor desconhecida"
            txt = f"ID:{id_curto} | Cor:{cor_txt}"

            if v_data["velocidade"]:
                txt += f" | {v_data['velocidade']} km/h"

                if v_data["contramao"]:
                    txt += " | !! CONTRAMAO !!"
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                else:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
            else:
                txt += " | A medir..."

            obj_meta.text_params.display_text = txt
            l_obj = l_obj.next

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


def main():
    init_cv2()
    Gst.init(None)

    pipeline_str = (
        "uridecodebin uri=file:///data/traffic.mp4 ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM), format=NV12 ! "
        "mux.sink_0 nvstreammux name=mux batch-size=1 width=1920 height=1080 ! "
        "nvinfer config-file-path=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt ! "
        "nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        "ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml "
        "tracker-width=640 tracker-height=384 ! "
        "nvdsanalytics config-file=/data/config_analytics.txt ! "
        "nvvideoconvert ! nvdsosd name=osd ! nveglglessink sync=1"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    osd = pipeline.get_by_name("osd")

    if not osd:
        print("ERRO: não encontrei o elemento OSD no pipeline.")
        return

    sink_pad = osd.get_static_pad("sink")

    if not sink_pad:
        print("ERRO: não encontrei o sink pad do OSD.")
        return

    sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)

    print("A monitorizar tráfego... Velocidade, contramão e cor ativos na consola/display.")
    print("Cores possíveis: branco, preto, cinzento/prata, vermelho, azul, amarelo/laranja, cor desconhecida")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        if cv_cap is not None:
            cv_cap.release()


if __name__ == "__main__":
    main()
