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
CONFIG_PGIE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt"
CONFIG_ANALYTICS_PATH = "/data/config_analytics.txt"
TRACKER_CONFIG_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
TRACKER_LIB_PATH = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

DISTANCE_METERS = 16.0

# DeepStream bounding boxes are expressed in the nvstreammux coordinate space.
MUX_W = 1920
MUX_H = 1080

vehicles = {}
cv_cap = None
cv_lock = threading.Lock()
frame_cache = {"num": -1, "frame": None}


def init_cv2():
    global cv_cap

    if cv2 is None:
        print("[WARNING] OpenCV/cv2 is not available. The color will be shown as 'unknown color'.")
        return

    cv_cap = cv2.VideoCapture(VIDEO_PATH)

    if not cv_cap.isOpened():
        print(f"[WARNING] Could not open {VIDEO_PATH} with OpenCV. The color will be shown as 'unknown color'.")
        cv_cap = None
        return

    width = int(cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cv_cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] OpenCV enabled for color estimation. Video: {width}x{height} @ {fps:.2f} FPS")


def get_cv2_frame(frame_num):
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


def get_object_crop(frame_bgr, obj_meta):
    if frame_bgr is None:
        return None

    try:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale_x = frame_w / float(MUX_W)
        scale_y = frame_h / float(MUX_H)

        rect = obj_meta.rect_params
        x1 = int(max(0, rect.left * scale_x))
        y1 = int(max(0, rect.top * scale_y))
        x2 = int(min(frame_w, (rect.left + rect.width) * scale_x))
        y2 = int(min(frame_h, (rect.top + rect.height) * scale_y))

        if x2 <= x1 or y2 <= y1:
            return None

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            return None

        return frame_bgr[y1:y2, x1:x2]

    except Exception:
        return None


def get_vehicle_body_pixels(crop_bgr):
    """
    Attempts to isolate the vehicle body before classifying its color.
    Uses GrabCut when possible and falls back to a central crop region.
    """
    if crop_bgr is None or crop_bgr.size == 0 or cv2 is None:
        return None

    h, w = crop_bgr.shape[:2]
    if w < 30 or h < 30:
        return None

    # Useful vehicle region: avoids road pixels, lower wheel area, and side margins.
    roi = crop_bgr[
        int(h * 0.08):int(h * 0.72),
        int(w * 0.10):int(w * 0.90)
    ].copy()

    if roi.size == 0:
        return None

    roi_h, roi_w = roi.shape[:2]

    try:
        # Resize the ROI to speed up GrabCut.
        max_side = 180
        if max(roi_h, roi_w) > max_side:
            scale = max_side / float(max(roi_h, roi_w))
            small = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = roi

        small_h, small_w = small.shape[:2]
        if small_w < 20 or small_h < 20:
            return None

        mask = np.zeros((small_h, small_w), np.uint8)
        rect = (
            int(small_w * 0.08),
            int(small_h * 0.08),
            max(1, int(small_w * 0.84)),
            max(1, int(small_h * 0.84)),
        )
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

        if scale != 1.0:
            foreground = cv2.resize(foreground, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    except Exception:
        foreground = np.zeros((roi_h, roi_w), np.uint8)
        foreground[int(roi_h * 0.15):int(roi_h * 0.72), int(roi_w * 0.18):int(roi_w * 0.82)] = 255

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Remove vegetation and strong shadows.
    vegetation = (hue >= 35) & (hue <= 85) & (saturation > 55) & (value > 45)
    shadows = value < 25

    final_mask = (foreground > 0) & (~vegetation) & (~shadows)
    pixels = roi[final_mask]

    if pixels.shape[0] < 40:
        fallback = roi[
            int(roi_h * 0.15):int(roi_h * 0.68),
            int(roi_w * 0.20):int(roi_w * 0.80)
        ]
        if fallback.size == 0:
            return None
        pixels = fallback.reshape(-1, 3)

    return pixels


def classify_color(crop_bgr):
    """
    Classifies the vehicle color using the percentage of body pixels in each color range.
    Final classes: white, black, gray/silver, red, blue, yellow/orange.
    """
    if crop_bgr is None or crop_bgr.size == 0 or cv2 is None:
        return None

    try:
        pixels = get_vehicle_body_pixels(crop_bgr)
        if pixels is None or len(pixels) < 40:
            return None

        pixels = pixels.reshape(-1, 1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        hue = hsv[:, 0]
        saturation = hsv[:, 1]
        value = hsv[:, 2]
        total = len(hue)

        if total < 40:
            return None

        scores = {
            "white": np.sum((value > 170) & (saturation < 80)) / total,
            "black": np.sum(value < 55) / total,
            "gray/silver": np.sum((saturation < 60) & (value >= 55) & (value <= 205)) / total,
            "red": np.sum(((hue < 10) | (hue > 170)) & (saturation > 85) & (value > 60)) / total,
            "blue": np.sum((hue >= 90) & (hue <= 130) & (saturation > 75) & (value > 50)) / total,
            "yellow/orange": np.sum((hue >= 18) & (hue <= 38) & (saturation > 75) & (value > 70)) / total,
        }

        color, score = max(scores.items(), key=lambda x: x[1])

        # Separate thresholds are used to reduce false positives.
        if color in ("white", "black", "gray/silver"):
            if score < 0.26:
                return None
        else:
            if score < 0.16:
                return None

        return color

    except Exception:
        return None


def update_vehicle_color(vehicle_data, color):
    if color is None:
        return

    vehicle_data["color_votes"].append(color)

    if len(vehicle_data["color_votes"]) > 30:
        vehicle_data["color_votes"] = vehicle_data["color_votes"][-30:]

    vote_count = Counter(vehicle_data["color_votes"])
    most_common_color, votes = vote_count.most_common(1)[0]

    if votes >= 3:
        vehicle_data["color"] = most_common_color


def osd_sink_pad_buffer_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_list = batch_meta.frame_meta_list

    while frame_list:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        except StopIteration:
            break

        current_time = time.time()
        frame_num = int(frame_meta.frame_num)
        frame_bgr = None

        object_list = frame_meta.obj_meta_list

        while object_list:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(object_list.data)
            except StopIteration:
                break

            obj_id = obj_meta.object_id
            short_id = obj_id & 0xFFFF

            if obj_id not in vehicles:
                vehicles[obj_id] = {
                    "top_time": None,
                    "base_time": None,
                    "speed": None,
                    "wrong_way": False,
                    "side": None,
                    "alerted": False,
                    "color": None,
                    "color_votes": [],
                    "color_attempts": 0,
                }

            vehicle_data = vehicles[obj_id]

            if vehicle_data["color_attempts"] < 35:
                if frame_bgr is None:
                    frame_bgr = get_cv2_frame(frame_num)

                crop = get_object_crop(frame_bgr, obj_meta)
                color = classify_color(crop)

                vehicle_data["color_attempts"] += 1
                update_vehicle_color(vehicle_data, color)

            user_meta_list = obj_meta.obj_user_meta_list

            while user_meta_list:
                try:
                    user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
                except StopIteration:
                    break

                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                    try:
                        analytics = pyds.NvDsAnalyticsObjMeta.cast(user_meta.user_meta_data)
                    except AttributeError:
                        analytics = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)

                    status = analytics.roiStatus

                    if any(x in status for x in ["L_Topo", "roi-L_Topo"]):
                        if vehicle_data["top_time"] is None:
                            vehicle_data["top_time"] = current_time
                            vehicle_data["side"] = "L"

                    elif any(x in status for x in ["R_Topo", "roi-R_Topo"]):
                        if vehicle_data["top_time"] is None:
                            vehicle_data["top_time"] = current_time
                            vehicle_data["side"] = "R"

                    if any(x in status for x in ["L_Base", "roi-L_Base"]):
                        if vehicle_data["base_time"] is None:
                            vehicle_data["base_time"] = current_time
                            vehicle_data["side"] = "L"

                    elif any(x in status for x in ["R_Base", "roi-R_Base"]):
                        if vehicle_data["base_time"] is None:
                            vehicle_data["base_time"] = current_time
                            vehicle_data["side"] = "R"

                    if vehicle_data["speed"] is None and vehicle_data["top_time"] and vehicle_data["base_time"]:
                        delta_time = abs(vehicle_data["top_time"] - vehicle_data["base_time"])

                        if delta_time > 0.3:
                            vehicle_data["speed"] = int((DISTANCE_METERS / delta_time) * 3.6)

                            if (vehicle_data["side"] == "L" and vehicle_data["base_time"] < vehicle_data["top_time"]) or \
                               (vehicle_data["side"] == "R" and vehicle_data["top_time"] < vehicle_data["base_time"]):
                                vehicle_data["wrong_way"] = True

                            color_text = vehicle_data["color"] or "unknown color"

                            if vehicle_data["wrong_way"] and not vehicle_data["alerted"]:
                                print(
                                    f"\033[91m[CRITICAL ALERT]\033[0m "
                                    f"Vehicle ID:{short_id} ({color_text}) detected driving WRONG WAY "
                                    f"at {vehicle_data['speed']} km/h!"
                                )
                                vehicle_data["alerted"] = True

                            elif not vehicle_data["wrong_way"] and not vehicle_data["alerted"]:
                                print(
                                    f"[INFO] Vehicle ID:{short_id} ({color_text}) passed correctly "
                                    f"at {vehicle_data['speed']} km/h."
                                )
                                vehicle_data["alerted"] = True

                user_meta_list = user_meta_list.next

            color_text = vehicle_data["color"] or "unknown color"
            display_text = f"ID:{short_id} | Color:{color_text}"

            if vehicle_data["speed"]:
                display_text += f" | {vehicle_data['speed']} km/h"

                if vehicle_data["wrong_way"]:
                    display_text += " | !! WRONG WAY !!"
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)
                else:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
            else:
                display_text += " | Measuring..."

            obj_meta.text_params.display_text = display_text
            object_list = object_list.next

        frame_list = frame_list.next

    return Gst.PadProbeReturn.OK


def main():
    init_cv2()
    Gst.init(None)

    pipeline_str = (
        f"uridecodebin uri=file://{VIDEO_PATH} ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM), format=NV12 ! "
        "mux.sink_0 nvstreammux name=mux batch-size=1 width=1920 height=1080 ! "
        f"nvinfer config-file-path={CONFIG_PGIE_PATH} ! "
        f"nvtracker ll-lib-file={TRACKER_LIB_PATH} "
        f"ll-config-file={TRACKER_CONFIG_PATH} "
        "tracker-width=640 tracker-height=384 ! "
        f"nvdsanalytics config-file={CONFIG_ANALYTICS_PATH} ! "
        "nvvideoconvert ! nvdsosd name=osd ! nveglglessink sync=1"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    osd = pipeline.get_by_name("osd")

    if not osd:
        print("ERROR: could not find the OSD element in the pipeline.")
        return

    sink_pad = osd.get_static_pad("sink")

    if not sink_pad:
        print("ERROR: could not find the OSD sink pad.")
        return

    sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)

    print("Monitoring traffic... Speed, wrong-way detection, and color estimation are active in the console/display.")
    print("Possible colors: white, black, gray/silver, red, blue, yellow/orange, unknown color")

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
