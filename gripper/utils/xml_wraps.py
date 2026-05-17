import numpy as np
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from robosuite.models.objects import MujocoXMLObject


class RobosuiteXMLAdapter(MujocoXMLObject):
	"""
	Adapter for raw MuJoCo XML assets that need to work with robosuite.

	This class loads an existing object XML, optionally normalizes it into a
	robosuite-friendly layout, and then delegates to
	robosuite.models.objects.MujocoXMLObject for parsing. The compatibility
	pass currently handles common issues such as:

	- wrapping direct object bodies in the structure robosuite expects
	- adding missing inertial and placement sites
	- normalizing geom groups and contact attributes
	- converting MuJoCo 1.x <freejoint> tags into MuJoCo 2.x <joint type="free"> tags

	Args:
		name: Object name exposed to robosuite.
		xml_path: Path to the source MuJoCo XML file.
		joints: Joint specification passed through to MujocoXMLObject.
		obj_type: Which geom groups to keep (collision, visual, or all).
		duplicate_collision_geoms: Whether to duplicate collision geoms for visuals.
		scale: Optional scale factor forwarded to robosuite.
		mass: Fallback mass used when the source XML is missing inertial data.
		diaginertia: Fallback diagonal inertia used when the source XML is missing inertial data.
		bottom_site: Placement site inserted on the outer body.
		top_site: Placement site inserted on the outer body.
		horizontal_radius_site: Placement site inserted on the outer body.
		auto_fix: If True, rewrite the XML into a robosuite-compatible cached copy.
	"""

	def __init__(
		self,
		name,
		xml_path,
		joints="default",
		obj_type="all",
		duplicate_collision_geoms=False,
		scale=None,
		mass=0.1,
		diaginertia=(0.001, 0.001, 0.001),
		bottom_site=(0.0, 0.0, 0.0),
		top_site=(0.0, 0.0, 0.10),
		horizontal_radius_site=(0.05, 0.05, 0.10),
		auto_fix=True,
	):
		xml_path = os.path.abspath(xml_path)

		if auto_fix:
			xml_path = self._make_robosuite_compatible_xml(
				xml_path=xml_path,
				mass=mass,
				diaginertia=diaginertia,
				bottom_site=bottom_site,
				top_site=top_site,
				horizontal_radius_site=horizontal_radius_site,
			)

		super().__init__(
			fname=xml_path,
			name=name,
			joints=joints,
			obj_type=obj_type,
			duplicate_collision_geoms=duplicate_collision_geoms,
			scale=scale,
		)

	@staticmethod
	def _vec_to_string(v):
		return " ".join(str(float(x)) for x in v)

	@classmethod
	def _make_robosuite_compatible_xml(
		cls,
		xml_path,
		mass,
		diaginertia,
		bottom_site,
		top_site,
		horizontal_radius_site,
	):
		xml_path = Path(xml_path)
		ycb_root = (
			xml_path.parent.parent
			if xml_path.parent.name in {"mj_xml", "_mj_xml", "robosuite_xml", "_robosuite_xml"}
			else xml_path.parent
		)
		tree = ET.parse(xml_path)
		root = tree.getroot()

		worldbody = root.find("worldbody")
		if worldbody is None:
			raise ValueError("XML has no <worldbody> tag.")

		bodies = list(worldbody.findall("body"))
		if len(bodies) == 0:
			raise ValueError("XML <worldbody> has no <body>.")

	   
		outer_body = bodies[0]
		object_body = outer_body.find("body[@name='object']")

		if object_body is None:
			# The XML is probably in the simple/direct format.
			# Treat the first body as the actual object body.
			original_body = outer_body

			# Remove existing bodies from worldbody.
			for b in bodies:
				worldbody.remove(b)

			# Create new outer robosuite body.
			outer_body = ET.Element("body")

			# Rename original body to the exact name robosuite expects.
			original_body.set("name", "object")

			object_body = original_body
			outer_body.append(object_body)
			worldbody.append(outer_body)

		# ------------------------------------------------------------
		# 2. Add inertial if missing
		# ------------------------------------------------------------

		inertial = object_body.find("inertial")
		if inertial is None:
			inertial = ET.Element(
				"inertial",
				attrib={
					"pos": "0 0 0",
					"mass": str(float(mass)),
					"diaginertia": cls._vec_to_string(diaginertia),
				},
			)
			object_body.insert(0, inertial)

		# ------------------------------------------------------------
		# 3. Fix geom groups
		#
		# robosuite convention:
		#   group="0" -> collision
		#   group="1" -> visual
		# ------------------------------------------------------------

		for geom in object_body.findall(".//geom"):
			mesh_name = geom.get("mesh", "")

			is_visual = (
				mesh_name == "model"
				or geom.get("material") is not None
				or geom.get("contype") == "0"
			)

			if is_visual:
				geom.set("group", "1")
				geom.set("contype", "0")
				geom.set("conaffinity", "0")
			else:
				geom.set("group", "0")
				geom.set("contype", "1")
				geom.set("conaffinity", "1")

				# Optional but useful defaults for manipulation objects
				if geom.get("friction") is None:
					geom.set("friction", "1 0.005 0.0001")
				if geom.get("condim") is None:
					geom.set("condim", "4")

		# ------------------------------------------------------------
		# 3.1. Rebase relative asset file paths for the moved XML layout
		# ------------------------------------------------------------

		for element in root.iter():
			file_path = element.get("file")
			if file_path and not Path(file_path).is_absolute() and not file_path.startswith("../"):
				element.set("file", f"../{file_path}")

		# ------------------------------------------------------------
		# 3.5. Convert freejoint to joint (MuJoCo 1.x to 2.x compatibility)
		# ------------------------------------------------------------

		for freejoint in object_body.findall("freejoint"):
			# Convert <freejoint name="..."> to <joint name="..." type="free">
			joint = ET.Element(
				"joint",
				attrib={
					"name": freejoint.get("name", "free_joint"),
					"type": "free",
				},
			)
			# Copy any other attributes (except name)
			for attr, value in freejoint.attrib.items():
				if attr != "name":
					joint.set(attr, value)
			# Insert joint at the same position as freejoint
			object_body.insert(list(object_body).index(freejoint), joint)
			object_body.remove(freejoint)

		# ------------------------------------------------------------
		# 4. Add robosuite placement sites to OUTER body
		# ------------------------------------------------------------

		def ensure_site(site_name, pos, size="0.005"):
			existing = outer_body.find(f"site[@name='{site_name}']")
			if existing is None:
				ET.SubElement(
					outer_body,
					"site",
					attrib={
						"name": site_name,
						"pos": cls._vec_to_string(pos),
						"size": size,
						"rgba": "0 0 0 0",
					},
				)

		ensure_site("bottom_site", bottom_site)
		ensure_site("top_site", top_site)
		ensure_site("horizontal_radius_site", horizontal_radius_site)

		# ------------------------------------------------------------
		# 5. Save fixed XML next to original XML
		#
		# Important: save next to original file so relative paths like
		# texture.png, model.obj, model_collision_0.obj still work.
		# ------------------------------------------------------------

		fixed_dir = ycb_root / "_robosuite_xml"
		fixed_dir.mkdir(parents=True, exist_ok=True)
		fixed_path = fixed_dir / f"{xml_path.stem}_robosuite.xml"

		ET.indent(tree, space="  ")
		tree.write(fixed_path, encoding="utf-8", xml_declaration=False)

		return str(fixed_path)
class YCBObject(RobosuiteXMLAdapter):
	"""
	Wrapper for YCB objects that loads from the original YCB XMLs but adds robosuite-compatible inertial and placement sites. 
 	This allows us to use the original YCB XMLs without modification while still supporting robosuite's conventions for object handling and placement. 
	Note: This class assumes the original YCB XMLs are in a directory structure like:
	ycb_root/
		_mj_xml/
			object_name.xml
		object_name/
			textured.obj
	---
	Args:
	object_name: Base name of the YCB object (e.g., "lemon", "bowl").
	instance_name: Optional unique name for this instance (e.g., "lemon_1"). If None, defaults to object_name.
	ycb_root: Root directory of the YCB dataset. If None, defaults to the "assets/ycb" directory relative to this file.
	joints: Joint specification passed through to MujocoXMLObject.
	obj_type: Which geom groups to keep (collision, visual, or all).
	duplicate_collision_geoms: Whether to duplicate collision geoms for visuals.
	scale: Optional scale factor forwarded to robosuite.
	"""

	def __init__(
		self,
		object_name,
		instance_name=None,
		ycb_root=None,
		joints=None,
		obj_type="all",
		duplicate_collision_geoms=False,
		scale=None,
	):
		ycb_root = self._resolve_ycb_root(ycb_root)
		self._base_name = object_name
		instance_name = instance_name if instance_name is not None else object_name

		xml_path = ycb_root / "_mj_xml" / f"{object_name}.xml"
		if not xml_path.exists():
			legacy_xml_path = ycb_root / f"{object_name}.xml"
			if legacy_xml_path.exists():
				xml_path = legacy_xml_path
			else:
				raise FileNotFoundError(f"YCB object XML not found: {xml_path}")

		obj_path = ycb_root / object_name / "textured.obj"
		if not obj_path.exists():
			raise FileNotFoundError(f"YCB visual mesh not found: {obj_path}")

		self._bounding_box_half_size = self._compute_obj_half_size(obj_path)
		self._bottom_offset = np.array([0.0, 0.0, -float(self._bounding_box_half_size[2])])
		self._top_offset = np.array([0.0, 0.0, float(self._bounding_box_half_size[2])])

		super().__init__(
			name=instance_name,
			xml_path=str(xml_path),
			joints=joints,
			obj_type=obj_type,
			duplicate_collision_geoms=duplicate_collision_geoms,
			scale=scale,
			auto_fix=True,
		)

	def exclude_from_prefixing(self, inp):
		if isinstance(inp, str):
			return inp.startswith(self.naming_prefix)

		if hasattr(inp, "get"):
			return inp.get("name", "").startswith(self.naming_prefix)

		return False
	
	@staticmethod
	def _resolve_ycb_root(ycb_root):
		if ycb_root is None:
			return Path(__file__).resolve().parents[1] / "assets" / "ycb"

		ycb_root = Path(ycb_root)
		if ycb_root.name in {"_mj_xml", "_robosuite_xml"}:
			return ycb_root.parent
		return ycb_root

	@staticmethod
	def _compute_obj_half_size(obj_path):
		mins = np.array([np.inf, np.inf, np.inf], dtype=float)
		maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

		with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				if line.startswith("v "):
					_, xs, ys, zs = line.split()[:4]
					vertex = np.array([float(xs), float(ys), float(zs)], dtype=float)
					mins = np.minimum(mins, vertex)
					maxs = np.maximum(maxs, vertex)

		return 0.5 * (maxs - mins)

	@property
	def bottom_offset(self):
		return self._bottom_offset

	@property
	def top_offset(self):
		return self._top_offset

	@property
	def horizontal_radius(self):
		return float(max(self._bounding_box_half_size[0], self._bounding_box_half_size[1]))

	def get_bounding_box_half_size(self):
		return self._bounding_box_half_size
	

