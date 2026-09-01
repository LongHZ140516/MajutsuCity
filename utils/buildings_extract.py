import os
import cv2
import json
import skimage
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from utils.function import extract_lmask

@dataclass
class ObjectInfo:
    id: int
    label: str
    mask: np.ndarray
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    height: float

def extract_objects_info(label_name: str, mask: np.ndarray, height_map: np.ndarray)-> Tuple[np.ndarray, int, List[ObjectInfo]]:
    """
    Extract the information of the objects in the mask.

    Args:
        label_name: str, the name of the label
        mask: np.ndarray, the mask of the image
        height_map: np.ndarray, the height map of the image
    Returns:
        divided_mask: np.ndarray, the divided mask of the image
        object_num: int, the number of the objects in the mask
        objects_info: List[ObjectInfo], the information of the objects in the mask
    """
    label_mask = extract_lmask(label_name, mask)

    divided_mask, object_num = skimage.measure.label(label_mask, connectivity=2, background=0, return_num=True)
    objects_info = []

    for i in range(object_num):
        object_mask = divided_mask == i + 1
        object_mask = object_mask.astype(np.uint8)
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_height = np.mean(height_map[object_mask > 0])

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 3:
            continue

        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect

        objects_info.append(ObjectInfo(
            id=i,
            label=label_name,
            mask=object_mask,
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            angle=angle,
            height=object_height
        ))
    
    return divided_mask, object_num, objects_info

def _rect_to_polygon_coords(cx: float, cy: float, w: float, h: float, angle: float,
                            decimals: Optional[int] = 2) -> List[List[float]]:
    """
    convert the minAreaRect parameters to the closed polygon coordinates [[x,y],...,[x,y]] (pixel coordinates; x=col,y=row).
    """
    rect = ((float(cx), float(cy)), (float(w), float(h)), float(angle))
    box = cv2.boxPoints(rect).tolist()  # 4x2
    box.append(box[0])                  # 闭合
    if decimals is not None:
        box = [[round(x, decimals), round(y, decimals)] for x, y in box]
    return box

def export_objects_info(
    objects: List[ObjectInfo],
    out_path: str,
    crs_name: str = "IMAGE_PIXEL",
    decimals: Optional[int] = 2
) -> None:
    """
    export the objects to the GeoJSON file, which includes the polygon and the centroid.

    Args:
        objects: List[ObjectInfo], the objects info
        out_path: str, the output path
        crs_name: str, the CRS name
        decimals: Optional[int], the decimal places

    """

    fc = {
        "type": "FeatureCollection",
        "name": "objects_poly_centroid",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": []
    }

    for obj in objects:
        # filter the abnormal objects
        if obj.w <= 0 or obj.h <= 0 or np.isnan(obj.cx) or np.isnan(obj.cy):
            continue

        poly_coords = _rect_to_polygon_coords(obj.cx, obj.cy, obj.w, obj.h, obj.angle, decimals=decimals)
        cx = float(obj.cx)
        cy = float(obj.cy)
        if decimals is not None:
            cx = round(cx, decimals)
            cy = round(cy, decimals)

        base_props = {
            "id": int(obj.id),
            "label": str(obj.label),
            "cx": float(obj.cx), "cy": float(obj.cy),
            "w": float(obj.w),   "h": float(obj.h),
            "angle": float(obj.angle)
        }

        
        poly_feat = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [poly_coords]},
            "properties": base_props | {"feat_type": "polygon"}
        }
        pt_feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [cx, cy]},
            "properties": base_props | {"feat_type": "centroid"}
        }
        fc["features"].extend([poly_feat, pt_feat])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def _box_points_from_cxcywh_angle(cx, cy, w, h, angle_deg):
    """get the 4 points of the box from the center point + width + height + angle"""
    a = np.deg2rad(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, -sa],
                  [sa,  ca]], dtype=np.float32)
    hw, hh = float(w) / 2.0, float(h) / 2.0
    rect_local = np.array([[-hw, -hh],
                           [ hw, -hh],
                           [ hw,  hh],
                           [-hw,  hh]], dtype=np.float32)
    pts = rect_local @ R.T + np.array([float(cx), float(cy)], dtype=np.float32)
    return pts 

def _color_from_label(label):
    """assign the color to the label"""
    h = hash(str(label)) & 0xFFFFFF
    b = (h      ) & 255
    g = (h >> 8 ) & 255
    r = (h >> 16) & 255
    return int(b), int(g), int(r)

def draw_oriented_bboxes(
    base_img: np.ndarray,
    objects_info: List[ObjectInfo],
    thickness: int = 2,
    with_text: bool = True,
    alpha_layer: float = 0.0, 
):
    """
    base_img: HxW or HxWx3 (BGR) base image
    objects_info: List[ObjectInfo]
    Returns: the image with the annotated objects
    """
    RED = (0, 0, 255)
    if base_img.ndim == 2:
        canvas = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_img.copy()

    if alpha_layer and alpha_layer > 0:
        layer = np.zeros_like(canvas)
        target = layer
    else:
        target = canvas

    for obj in objects_info:
        cx, cy = float(obj.cx), float(obj.cy)
        w, h   = float(obj.w),  float(obj.h)
        angle  = float(obj.angle)
        label  = obj.label
        id = obj.id

        pts   = _box_points_from_cxcywh_angle(cx, cy, w, h, angle)
        pts_i = np.round(pts).astype(np.int32)

        # line box
        cv2.polylines(target, [pts_i], isClosed=True, color=RED, thickness=thickness, lineType=cv2.LINE_AA)
        # center point
        cv2.circle(target, (int(round(cx)), int(round(cy))), radius=2, color=RED, thickness=-1, lineType=cv2.LINE_AA)

        if with_text:
            idx = np.lexsort((pts[:,0], pts[:,1]))[0]
            tl  = pts_i[idx]
            txt = f"{label}#{id}" if id is not None else str(label)
            cv2.putText(target, txt, (tl[0], tl[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)

    if alpha_layer and alpha_layer > 0:
        out = cv2.addWeighted(canvas, 1.0, layer, float(alpha_layer), 0.0)
    else:
        out = canvas 
    return out
