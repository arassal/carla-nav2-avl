"""Static checks on the /perception/reset handler.

`costmap_node.py` cannot be imported yet -- it references four modules that
were never committed (ISSUES.md B1) -- so the handler cannot be exercised at
runtime here. These read the source instead, which is enough to catch the
failure mode that actually threatens attribute-based reset code: assigning a
name that does not exist. Python creates it silently, the real attribute keeps
its stale value, and the reset looks like it worked.

Replace these with a live service call once B1 lands.
"""

import ast
import os

import pytest


SOURCE = os.path.join(os.path.dirname(__file__), os.pardir,
                      "perception_costmap", "costmap_node.py")


def _tree():
    with open(SOURCE) as handle:
        return ast.parse(handle.read())


def _self_attrs_assigned(func):
    """Names assigned as `self.<name> = ...` or via setattr(self, "<name>", ...)."""
    found = set()
    for node in ast.walk(func):
        for target in (node.targets if isinstance(node, ast.Assign) else []):
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                found.add(target.attr)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "setattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
                and isinstance(node.args[1], ast.Constant)):
            found.add(node.args[1].value)
    # names listed in a `for counter in (...)` loop driving setattr
    for node in ast.walk(func):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
            for element in node.iter.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    found.add(element.value)
    return found


def _method(class_name, method_name):
    for node in _tree().body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError("%s.%s not found" % (class_name, method_name))


def test_reset_only_touches_attributes_the_constructor_creates():
    created = _self_attrs_assigned(_method("CostmapNode", "__init__"))
    reset = _self_attrs_assigned(_method("CostmapNode", "_on_reset"))
    unknown = reset - created
    assert not unknown, (
        "reset assigns attributes never set in __init__: %s. Either it is a "
        "typo (Python creates the name silently and the real attribute keeps "
        "its stale value), or __init__ needs to initialise it too."
        % sorted(unknown))


def test_reset_clears_the_temporal_filters():
    """The filters are the only state that survives a tick by design."""
    source = ast.unparse(_method("CostmapNode", "_on_reset"))
    assert "obs_filters" in source and "reset()" in source


def test_reset_clears_every_accumulating_counter():
    """Diagnostics that only ever increment must restart at zero too."""
    reset = _self_attrs_assigned(_method("CostmapNode", "_on_reset"))
    for counter in ("_ticks", "_depth_projections", "_ipm_fallbacks",
                    "_detector_errors", "_depth_waits"):
        assert counter in reset, "%s is never reset" % counter


def test_camera_state_is_reset_for_every_camera():
    source = ast.unparse(_method("CostmapNode", "_on_reset"))
    assert "for cam in self.cameras" in source
    for attr in ("last_yolo_stamp", "last_cone_stamp",
                 "depth_buffer", "confidence_buffer"):
        assert attr in source, "camera %s is never reset" % attr


def test_service_is_registered_with_the_documented_name():
    source = ast.unparse(_method("CostmapNode", "__init__"))
    assert "/perception/reset" in source
    assert "_on_reset" in source
