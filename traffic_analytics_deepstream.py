import os
import time

import gi

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds


VIDEO_PATH = "/data/traffic.mp4"

CONFIG_PGIE_PATH = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt"
CONFIG_ANALYTICS_PATH = "/data/config_analytics.txt"

# Official/sample Secondary GIE configuration for vehicle type classification.
# If this file does not exist inside the container, the application will run
# without vehicle type classification.
CONFIG_SGIE_VEHICLE_TYPE_PATH = "/data/config_infer_secondary_vehicletypes_runtime.txt"

DISTANCE_METERS = 16.0

vehicles = {}

CLASS_LABELS_FALLBACK = {
    0: "Vehicle",
    1: "TwoWheeler",
    2: "Person",
    3: "RoadSign",
}

CLASS_LABELS = {}


def clean_label(label):
    if label is None:
        return ""

    try:
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="ignore")

        return str(label).split("\x00")[0].strip()

    except Exception:
        return ""


def resolve_relative_path(base_file, path):
    path = path.strip()

    if not path:
        return path

    if os.path.isabs(path):
        return os.path.abspath(path)

    return os.path.abspath(os.path.join(os.path.dirname(base_file), path))


def load_labels_from_config(config_path):
    labels = {}

    try:
        labelfile_path = None

        with open(config_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if line.startswith("labelfile-path="):
                    labelfile_path = line.split("=", 1)[1].strip()
                    break

        if labelfile_path:
            labelfile_path = resolve_relative_path(config_path, labelfile_path)

            if os.path.exists(labelfile_path):
                with open(labelfile_path, "r", encoding="utf-8", errors="ignore") as file:
                    for idx, line in enumerate(file):
                        label = line.strip()
                        if label:
                            labels[idx] = label

                if labels:
                    print(f"[INFO] PGIE labels loaded from: {labelfile_path}")
                    print(f"[INFO] PGIE classes: {labels}")
                    return labels

    except Exception as error:
        print(f"[WARNING] Could not read PGIE labels: {error}")

    print(f"[WARNING] Using PGIE fallback classes: {CLASS_LABELS_FALLBACK}")
    return CLASS_LABELS_FALLBACK.copy()


def normalize_pgie_class(raw_label, class_id):
    label = (raw_label or "").strip()
    normalized = label.lower().replace("-", "").replace("_", "").replace(" ", "")

    if normalized in ("vehicle", "vehicles", "car", "cars", "automobile", "auto"):
        return "vehicle"
    if normalized in ("truck", "lorry"):
        return "truck"
    if normalized in ("bus",):
        return "bus"
    if normalized in ("motorbike", "motorcycle", "bike", "bicycle", "twowheeler", "twowheelers"):
        return "two_wheeler"
    if normalized in ("person", "persons", "pedestrian"):
        return "person"
    if normalized in ("roadsign", "sign", "trafficsign"):
        return "road_sign"
    if normalized:
        return label

    fallback = CLASS_LABELS.get(class_id, f"class_{class_id}")
    return normalize_pgie_class(fallback, -1) if class_id != -1 else fallback


def get_pgie_class(obj_meta):
    class_id = int(obj_meta.class_id)
    raw_label = clean_label(getattr(obj_meta, "obj_label", ""))

    if not raw_label:
        raw_label = CLASS_LABELS.get(class_id, f"class_{class_id}")

    pgie_class = normalize_pgie_class(raw_label, class_id)
    return class_id, raw_label, pgie_class


def is_vehicle_class(pgie_class):
    normalized = (pgie_class or "").lower()

    return normalized in {
        "vehicle",
        "truck",
        "bus",
        "two_wheeler",
        "car",
        "motorcycle",
        "bicycle",
        "bike",
        "twowheeler",
    }


def normalize_vehicle_type(label):
    """
    Normalizes labels returned by SGIE VehicleTypeNet or other classifiers.
    The list is intentionally permissive so it can work with different label files.
    """
    raw_label = clean_label(label)
    normalized = raw_label.lower().replace("-", "").replace("_", "").replace(" ", "")

    if not normalized:
        return None

    if normalized in ("car", "cars", "sedan", "coupe", "hatchback", "suv", "vehicle", "automobile"):
        return "car"

    if normalized in ("van", "minivan", "pickup", "pickuptruck"):
        return "van"

    if normalized in ("truck", "lorry", "largevehicle", "largevehicles", "heavytruck", "semi", "trailer"):
        return "truck"

    if normalized in ("bus", "coach"):
        return "bus"

    if normalized in ("motorbike", "motorcycle", "moto", "bike", "bicycle", "twowheeler", "twowheelers"):
        return "motorcycle/two_wheeler"

    return raw_label


def get_sgie_vehicle_type(obj_meta):
    """
    Reads the secondary classifier result from the object metadata.
    Returns (normalized_vehicle_type, probability).
    """
    best_type = None
    best_probability = 0.0

    classifier_list = obj_meta.classifier_meta_list

    while classifier_list:
        try:
            classifier_meta = pyds.NvDsClassifierMeta.cast(classifier_list.data)
        except StopIteration:
            break

        label_list = classifier_meta.label_info_list

        while label_list:
            try:
                label_info = pyds.NvDsLabelInfo.cast(label_list.data)
            except StopIteration:
                break

            label = clean_label(getattr(label_info, "result_label", ""))

            try:
                probability = float(label_info.result_prob)
            except Exception:
                probability = 0.0

            vehicle_type = normalize_vehicle_type(label)

            if vehicle_type and probability >= best_probability:
                best_type = vehicle_type
                best_probability = probability

            label_list = label_list.next

        classifier_list = classifier_list.next

    return best_type, best_probability


def update_vehicle_type(vehicle_data, vehicle_type, probability):
    """
    Uses temporal voting to stabilize the vehicle type.
    This avoids switching between car/truck/bus/etc. due to isolated readings.
    """
    if not vehicle_type:
        return

    # If the model does not provide a useful probability, use weight 1.
    weight = probability if probability and probability > 0 else 1.0

    vehicle_data["type_scores"][vehicle_type] = vehicle_data["type_scores"].get(vehicle_type, 0.0) + weight
    vehicle_data["type_votes"][vehicle_type] = vehicle_data["type_votes"].get(vehicle_type, 0) + 1

    top_type = max(vehicle_data["type_scores"], key=vehicle_data["type_scores"].get)
    top_votes = vehicle_data["type_votes"].get(top_type, 0)

    # Only accept the type after a few consistent votes.
    if top_votes >= 2:
        vehicle_data["type"] = top_type


def vehicle_type_text(vehicle_data):
    return vehicle_data.get("type") or "unknown type"


def osd_sink_pad_buffer_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    frame_list = batch_meta.frame_meta_list

    while frame_list:
        frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        object_list = frame_meta.obj_meta_list
        current_time = time.time()

        while object_list:
            obj_meta = pyds.NvDsObjectMeta.cast(object_list.data)

            object_id = obj_meta.object_id
            short_id = object_id & 0xFFFF

            class_id, raw_class, pgie_class = get_pgie_class(obj_meta)
            is_vehicle = is_vehicle_class(pgie_class)

            if object_id not in vehicles:
                vehicles[object_id] = {
                    "top_time": None,
                    "base_time": None,
                    "speed": None,
                    "wrong_way": False,
                    "side": None,
                    "alerted": False,
                    "pgie_class": pgie_class,
                    "raw_class": raw_class,
                    "class_id": class_id,
                    "type": None,
                    "type_scores": {},
                    "type_votes": {},
                }
            else:
                vehicles[object_id]["pgie_class"] = pgie_class
                vehicles[object_id]["raw_class"] = raw_class
                vehicles[object_id]["class_id"] = class_id

            vehicle_data = vehicles[object_id]

            # Read the vehicle type from the secondary classifier, when available.
            if is_vehicle:
                vehicle_type, probability = get_sgie_vehicle_type(obj_meta)
                update_vehicle_type(vehicle_data, vehicle_type, probability)

            # If the object is not a vehicle, show its class but do not measure speed.
            if not is_vehicle:
                obj_meta.text_params.display_text = f"ID:{short_id} | Class:{pgie_class}"
                object_list = object_list.next
                continue

            user_meta_list = obj_meta.obj_user_meta_list

            while user_meta_list:
                user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)

                if user_meta.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                    try:
                        analytics = pyds.NvDsAnalyticsObjMeta.cast(user_meta.user_meta_data)
                    except AttributeError:
                        analytics = pyds.NvDsAnalyticsObjInfo.cast(user_meta.user_meta_data)

                    status = analytics.roiStatus

                    # Register ROI crossings.
                    if any(roi in status for roi in ["L_Topo", "roi-L_Topo"]):
                        if vehicle_data["top_time"] is None:
                            vehicle_data["top_time"] = current_time
                            vehicle_data["side"] = "L"

                    elif any(roi in status for roi in ["R_Topo", "roi-R_Topo"]):
                        if vehicle_data["top_time"] is None:
                            vehicle_data["top_time"] = current_time
                            vehicle_data["side"] = "R"

                    if any(roi in status for roi in ["L_Base", "roi-L_Base"]):
                        if vehicle_data["base_time"] is None:
                            vehicle_data["base_time"] = current_time
                            vehicle_data["side"] = "L"

                    elif any(roi in status for roi in ["R_Base", "roi-R_Base"]):
                        if vehicle_data["base_time"] is None:
                            vehicle_data["base_time"] = current_time
                            vehicle_data["side"] = "R"

                    # Speed calculation and alert generation.
                    if vehicle_data["speed"] is None and vehicle_data["top_time"] and vehicle_data["base_time"]:
                        delta_time = abs(vehicle_data["top_time"] - vehicle_data["base_time"])

                        if delta_time > 0.3:
                            vehicle_data["speed"] = int((DISTANCE_METERS / delta_time) * 3.6)

                            # Determine whether the vehicle is driving the wrong way.
                            if (vehicle_data["side"] == "L" and vehicle_data["base_time"] < vehicle_data["top_time"]) or \
                               (vehicle_data["side"] == "R" and vehicle_data["top_time"] < vehicle_data["base_time"]):
                                vehicle_data["wrong_way"] = True

                            type_text = vehicle_type_text(vehicle_data)

                            if vehicle_data["wrong_way"] and not vehicle_data["alerted"]:
                                print(
                                    f"\033[91m[CRITICAL ALERT]\033[0m "
                                    f"{type_text} ID:{short_id} detected driving the WRONG WAY "
                                    f"at {vehicle_data['speed']} km/h!"
                                )
                                vehicle_data["alerted"] = True

                            elif not vehicle_data["wrong_way"] and not vehicle_data["alerted"]:
                                print(
                                    f"[INFO] {type_text} ID:{short_id} passed correctly "
                                    f"at {vehicle_data['speed']} km/h."
                                )
                                vehicle_data["alerted"] = True

                user_meta_list = user_meta_list.next

            # Video overlay.
            type_text = vehicle_type_text(vehicle_data)
            display_text = f"ID:{short_id} | Type:{type_text}"

            if vehicle_data["speed"]:
                display_text += f" | {vehicle_data['speed']} km/h"

                if vehicle_data["wrong_way"]:
                    display_text += " | !! WRONG WAY !!"
                    obj_meta.rect_params.border_color.set(1.0, 0.0, 0.0, 1.0)  # Red
                else:
                    obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)  # Green
            else:
                display_text += " | Measuring..."

            obj_meta.text_params.display_text = display_text

            object_list = object_list.next

        frame_list = frame_list.next

    return Gst.PadProbeReturn.OK


def main():
    global CLASS_LABELS

    CLASS_LABELS = load_labels_from_config(CONFIG_PGIE_PATH)

    if not os.path.exists(CONFIG_SGIE_VEHICLE_TYPE_PATH):
        print(f"[WARNING] SGIE config not found: {CONFIG_SGIE_VEHICLE_TYPE_PATH}")
        print("[WARNING] The application will run without vehicle type classification; type will be shown as 'unknown type'.")
        use_sgie = False
    else:
        print(f"[INFO] SGIE VehicleType enabled: {CONFIG_SGIE_VEHICLE_TYPE_PATH}")
        use_sgie = True

    Gst.init(None)

    if use_sgie:
        sgie_part = f"nvinfer name=sgie_vehicle_type config-file-path={CONFIG_SGIE_VEHICLE_TYPE_PATH} ! "
    else:
        sgie_part = ""

    pipeline_str = (
        f"uridecodebin uri=file://{VIDEO_PATH} ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM), format=NV12 ! "
        "mux.sink_0 nvstreammux name=mux batch-size=1 width=1920 height=1080 ! "
        f"nvinfer name=pgie config-file-path={CONFIG_PGIE_PATH} ! "
        "nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so "
        "ll-config-file=/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml "
        "tracker-width=640 tracker-height=384 ! "
        f"{sgie_part}"
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

    print("Monitoring traffic... Vehicle type, speed, and wrong-way detection are active in the console/display.")
    print("Note: car/truck/bus/motorcycle classification depends on the SGIE vehicle type model.")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
