import os
import cv2
import numpy as np
from PIL import Image
import pyvista as pv
from typing import Optional, Tuple, Dict, Any

from utils.buildings_extract import ObjectInfo

# -----------------------------
# Render Mesh to image
# -----------------------------
class RenderMesh:
    def __init__(self, 
            window_size: Tuple[int, int] = (1024, 1024), 
            mesh_color: Tuple[float, float, float] = (0.7, 0.7, 0.7), 
            background: Tuple[float, float, float] = (1.0, 1.0, 1.0), 
            transparent_background: bool = False, 
            show_edges: bool = False, 
            edge_color: Tuple[float, float, float] = (1, 1, 1), 
            line_width: float = 1.0):

        self.window_size: Tuple[int, int] = window_size
        self.mesh_color: Tuple[float, float, float] = mesh_color
        self.background: Tuple[float, float, float] = background
        self.transparent_background: bool = transparent_background
        self.show_edges: bool = show_edges
        self.edge_color: Tuple[float, float, float] = edge_color
        self.line_width: float = line_width

    def setup_offscreen_rendering(self):
        """
        setup the offscreen rendering
        """
        os.environ['PYVISTA_OFF_SCREEN'] = 'true'
        os.environ['PYVISTA_USE_PANEL'] = 'false'
        
        try:
            pv.set_plot_theme("default")
            pv.start_xvfb()
        except Exception:
            pass

    def polygon_from_mask(self, mask: np.ndarray,
                               simplify_tol: float = 0,
                               min_area_px: float = 1.0) -> np.ndarray:
        """
        extract the polygon from the mask

        Args:
            mask: the mask
            simplify_tol: the simplify tolerance
            min_area_px: the minimum area of the polygon
        Returns:
            the polygon
        """
        if mask.dtype != np.uint8:
            bin8 = (mask > 0).astype(np.uint8)
        else:
            bin8 = (mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(bin8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contour found in mask.")

        # extract the largest polygon
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        # if areas[idx] < min_area_px:
        #     raise ValueError("Mask area too small.")
        cnt = contours[idx]  # (N,1,2)

        # simplify the polygon
        if simplify_tol > 0:
            epsilon = simplify_tol
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
        pts = cnt.reshape(-1, 2).astype(np.float32)  # (N,2) -> (x,y)

        return pts
    
    def polygon_to_osm(self, xy: np.ndarray) -> Dict[str, Any]:
        """
        convert the polygon to the osm
        Args:
            xy: the polygon
        Returns:
            the osm
        """
        nodes = [{"id": i, "x": float(xy[i, 0]), "y": float(xy[i, 1])} for i in range(len(xy))]
        way = {
            "id": 0,
            "nodes": list(range(len(xy))),
            "tags": ["building"]
        }
        return {"nodes": nodes, "ways": [way]}
    
    def closed_ordered_ring(self, xy: np.ndarray) -> np.ndarray:
        """
        ensure the polygon is closed and the vertices are in CCW direction
        Args:
            xy: the polygon
        Returns:
            the closed polygon
        """
        if xy.shape[0] < 3:
            raise ValueError("Not enough vertices to form a polygon.")

        # close the polygon
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])

        signed_area = 0.5 * np.sum(
            xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1]
        )
        if signed_area < 0:
            xy = xy[::-1].copy()
            if not np.allclose(xy[0], xy[-1]):
                xy = np.vstack([xy, xy[0]])

        return xy
    
    def footprint_to_mesh(self, footprint_xy: np.ndarray,
                      height: float,
                      pixel_size: float = 1.0,
                      base_z: float = 0.0) -> pv.PolyData:
        """
        stretch the 2D footprint (pixel coordinate system) to a 3D geometry.
        Args:
            footprint_xy: (N,2) the vertices sequence (x,y), unit: pixel
            height:       the height in the vertical direction, unit: pixel_size
            pixel_size:   the ratio of the pixel to the real world unit (e.g. 0.2 m/px)
            base_z:       the base z of the geometry
        Returns:
            the 3D geometry
        """
        xy = self.closed_ordered_ring(np.asarray(footprint_xy, dtype=np.float32))

        # convert the coordinate system: the image y is downward, the 3D y is upward, so the y' = -y
        pts2d = np.column_stack([xy[:, 0] * pixel_size,
                                -xy[:, 1] * pixel_size,
                                np.full(xy.shape[0], base_z, dtype=np.float32)])

        # build a "face":
        n = pts2d.shape[0] - 1  # the last point is the same as the first point (closed), so the faces only need the first n points
        points = pts2d[:-1].copy()  # remove the repeated closed points
        faces = np.hstack([[n], np.arange(n, dtype=np.int32)]).astype(np.int32)

        poly = pv.PolyData(points, faces=faces)

        # stretch (along the z axis)
        extruded = poly.extrude([0.0, 0.0, float(height)], capping=True)
        
        return extruded
    
    def render_isometric(self, mesh: pv.PolyData,
                     ) -> Dict[str, np.ndarray]:
        """
        render the mesh to the image with the isometric view
        Args:
            mesh: the mesh
        Returns:
            the image
        """
        plotter = pv.Plotter(off_screen=True, window_size=self.window_size)
        plotter.set_background(self.background)
        plotter.enable_parallel_projection()         # parallel projection → isometric effect
        plotter.add_mesh(mesh,
                        color=self.mesh_color,
                        show_edges=self.show_edges,
                        edge_color=self.edge_color,
                        line_width=self.line_width)

        # view: isometric, and adapt to the bounding box
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()

        plotter.render()
        
        # print(f"Plotter window size: {plotter.window_size}")
        # print(f"Mesh bounds: {mesh.bounds}")
        
        # get the depth map
        try:
            plotter.show() # without this, the depth map is not correct
            depth = plotter.get_image_depth()
            # print(f"Raw depth info: shape={depth.shape}, dtype={depth.dtype}")
            # print(f"Depth range: min={np.nanmin(depth):.6f}, max={np.nanmax(depth):.6f}")
            # print(f"NaN count: {np.isnan(depth).sum()}, Unique values: {len(np.unique(depth[~np.isnan(depth)]))}")
                    
        except Exception as e:
            print(f"get_image_depth() failed: {e}")
        
        # get the RGB image
        rgb = plotter.screenshot(transparent_background=self.transparent_background)
        
        plotter.close()

        return {"rgb": rgb, "depth": depth.astype(np.float32)}
    
    def build_and_render_from_mask(self, mask: np.ndarray,
                               height: float,
                               pixel_size: float = 1.0,
                               simplify_tol: float = 0.8,
                               min_area_px: float = 32.0,
                               obj_info: Optional[ObjectInfo] = None,
                               export_osm: bool = False
                               ) -> Dict[str, Any]:
        """
        mask→outline→3D stretch→isometric rendering.

        Args:
            mask: the mask
            height: the height
            pixel_size: the pixel size
            simplify_tol: the simplify tolerance
            min_area_px: the minimum area of the polygon
            obj_info: the object information
            export_osm: whether to export the OSM style structure
        Returns:
            the result
            {
                "mesh": pv.PolyData,
                "footprint_xy": (N,2) ndarray,
                "osm": {...} or None,
                "rgb": np.uint8[H,W,3/4],
                "depth": np.float32[H,W]
            }
        """
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array with a single building.")

        # 1) extract the footprint (outline)
        footprint_xy = self.polygon_from_mask(mask, simplify_tol=simplify_tol, min_area_px=min_area_px)

        # 2) (optional) export the OSM structure
        osm_obj = self.polygon_to_osm(footprint_xy) if export_osm else None

        # 3) build the 3D geometry
        mesh = self.footprint_to_mesh(footprint_xy, height=float(height), pixel_size=float(pixel_size), base_z=0.0)

        # 4) isometric rendering (RGB + depth)
        render_out = self.render_isometric(mesh)

        return {
            "mesh": mesh,
            "footprint_xy": footprint_xy,
            "osm": osm_obj,
            "rgb": render_out["rgb"],
            "depth": render_out["depth"],
        }
    
    def save_render_image(self, render_out: Dict[str, Any], output_dir: str):
        """
        save the render image
        Args:
            render_out: the render output
            output_dir: the output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        rgb = render_out["rgb"]
        depth = render_out["depth"]

        # save the rgb image
        rgb_path = os.path.join(output_dir, "rgb.png")
        Image.fromarray(rgb).save(rgb_path)

        # save the depth image
        depth_path = os.path.join(output_dir, "depth.png")
        depth_clean = np.nan_to_num(depth, nan=0.0)
        depth_min, depth_max = np.min(depth_clean), np.max(depth_clean)
        if depth_max > depth_min:
            depth_normalized = (depth_clean - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.ones_like(depth_clean) * 0.5

        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        Image.fromarray(depth_uint8, mode='L').save(depth_path)
    
    def sample_points_from_vertices(self, surface, num_samples, seed=None):
        rng = np.random.default_rng(seed)
        n = surface.n_points
        if n == 0:
            raise ValueError("表面没有点，无法抽样。")
        if num_samples <= n:
            idx = rng.choice(n, size=num_samples, replace=False)
        else:
            # 先拿完全部唯一点，再用放回抽样补齐
            extra = rng.choice(n, size=num_samples - n, replace=True)
            idx = np.concatenate([np.arange(n), extra])
        return surface.points[idx]
    
    def sample_points_from_mesh(
        self,
        mesh: pv.PolyData,
        num_samples: int = 10000,
        with_normals: bool = True,
        seed: int | None = None,
        method: str = "barycentric",
    ) -> pv.PolyData:
        rng = np.random.default_rng(seed)
        surf = mesh.extract_surface().triangulate().clean()
        if surf.n_cells == 0:
            raise ValueError("输入网格没有可用三角面。")

        if method == "vertices":
            n = surf.n_points
            if n == 0:
                raise ValueError("表面没有点，无法抽样。")
            if num_samples <= n:
                idx = rng.choice(n, size=num_samples, replace=False)
            else:
                extra = rng.choice(n, size=num_samples - n, replace=True)
                idx = np.concatenate([np.arange(n), extra])
            samples = surf.points[idx]
            pcd = pv.PolyData(samples)
        else:
            faces_all = surf.faces.reshape(-1, 4)
            tri_mask = faces_all[:, 0] == 3
            if not np.any(tri_mask):
                raise ValueError("三角面不存在，无法采样。")
            faces = faces_all[tri_mask][:, 1:]
            V = surf.points
            v0, v1, v2 = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
            good = areas > 0
            if not np.any(good):
                raise ValueError("所有三角面面积为 0。")
            v0, v1, v2, areas = v0[good], v1[good], v2[good], areas[good]
            probs = areas / areas.sum()
            tri_idx = rng.choice(len(areas), size=num_samples, p=probs)
            a, b, c = v0[tri_idx], v1[tri_idx], v2[tri_idx]
            r1, r2 = rng.random(num_samples), rng.random(num_samples)
            sr1 = np.sqrt(r1)
            samples = (1. - sr1)[:, None] * a + (sr1 * (1. - r2))[:, None] * b + (sr1 * r2)[:, None] * c
            pcd = pv.PolyData(samples)

        if with_normals:
            surf_n = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
            # ✅ 关键：在“采样点”上插值
            pcd = pcd.interpolate(surf_n)
            if "Normals" in pcd.point_data:
                pcd.point_data.set_array(pcd.point_data["Normals"], "Normals")
        return pcd
