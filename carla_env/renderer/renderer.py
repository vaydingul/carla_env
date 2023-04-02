import cv2

from utils.path_utils import create_date_time_path
from .colors import COLORS
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class Renderer:
    """
    It is a generic class for rendering the environment that works with cv2.
    It has a cursor property to keep track of last rendered pixel.
    Cursor can go to left-top, right-top, left-bottom, right-bottom, center; after
    rendering an object.
    It can handle texts, images, and points.
    """

    def __init__(self, config):

        self.set_default_config()
        self.update_config(config)
        self.build_from_config()

        if self.create_date_time_path:
            self.save_path = create_date_time_path(self.save_path)
        else:
            self.save_path = Path(self.save_path)
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.reset()

    def build_from_config(self):

        self.width = self.config["width"]
        self.height = self.config["height"]
        self.channel = self.config["channel"]
        self.background_color = self.config["background_color"]
        self.font = self.config["font"]
        self.font_scale = self.config["font_scale"]
        self.font_color = self.config["font_color"]
        self.font_thickness = self.config["font_thickness"]
        self.cursor = self.config["cursor"]
        self.name = self.config["name"]
        self.show_ = self.config["show"]
        self.save_ = self.config["save"]
        self.create_date_time_path = self.config["create_date_time_path"]
        self.save_path = self.config["save_path"]

    def reset(self):
        self.canvas = np.zeros((self.height, self.width, self.channel), np.uint8)
        self.canvas[:] = self.background_color
        self.cursor = (0, 0)

    def render_image(self, img, move_cursor="right"):

        assert (
            img.shape[-1] == self.channel
        ), "Image channel must be equal to renderer channel"

        H, W, C = img.shape

        self.canvas[
            self.cursor[0] : self.cursor[0] + H, self.cursor[1] : self.cursor[1] + W
        ] = img

        self._move_cursor(direction=move_cursor, amount=(H, W))

    def render_overlay_image(
        self, img1, img2, alpha1=0.5, alpha2=0.5, move_cursor="right"
    ):

        H1, W1, C1 = img1.shape
        H2, W2, C2 = img2.shape

        assert (H1, W1, C1) == (H2, W2, C2), "Image shapes must be equal"

        assert (
            img1.shape[-1] == self.channel
        ), "Image channel must be equal to renderer channel"

        self.canvas[
            self.cursor[0] : self.cursor[0] + H1, self.cursor[1] : self.cursor[1] + W1
        ] = cv2.addWeighted(img1, alpha1, img2, alpha2, 0)

        self._move_cursor(direction=move_cursor, amount=(H1, W1))

    def render_text(
        self,
        text="",
        move_cursor="right",
        font=None,
        font_scale=None,
        font_color=None,
        font_thickness=None,
    ):

        if font is None:
            font = self.font
        if font_scale is None:
            font_scale = self.font_scale
        if font_color is None:
            font_color = self.font_color
        if font_thickness is None:
            font_thickness = self.font_thickness

        text_cursor_ = tuple(reversed(self.cursor))
        (TEXT_W, TEXT_H), BASELINE = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        cv2.putText(
            self.canvas,
            text,
            text_cursor_,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

        self._move_cursor(direction=move_cursor, amount=(TEXT_H + BASELINE, TEXT_W))

    def render_point(
        self, pos=None, color=None, radius=5, thickness=-1, move_cursor="none"
    ):

        if color is None:
            color = self.font_color

        if pos is None:
            pos = tuple(reversed(self.cursor))

        cv2.circle(
            self.canvas,
            pos,
            radius,
            color,
            thickness,
        )

        self._move_cursor(direction=move_cursor, amount=(0, 0))

    def render_arrow(
        self, start, end, color=None, thickness=1, tip_length=0.5, move_cursor="none"
    ):

        if color is None:
            color = self.font_color

        cv2.arrowedLine(
            self.canvas,
            start,
            end,
            color,
            thickness,
            tipLength=tip_length,
        )

        self._move_cursor(direction=move_cursor, amount=(0, 0))

    def resize(self, fx=None, fy=None, width=None, height=None):

        assert ((fx is not None) and (fy is not None)) or (
            (width is not None) and (height is not None)
        ), "At least (fx ,fy) or (width, height) must be given"

        if fx is not None and fy is not None:
            self.canvas = cv2.resize(self.canvas, dsize=(0, 0), fx=fx, fy=fy)
        elif width is not None and height is not None:
            self.canvas = cv2.resize(self.canvas, (width, height))

    def show(self):

        if self.show_:
            cv2.imshow(f"{self.name}", self.canvas)
            cv2.waitKey(1)

    def save(self, info=None, ext="png"):

        if info is not None:
            save_path = f"{self.save_path}/{info}.{ext}"
        else:
            save_path = f"{self.save_path}/rendered_image.{ext}"

        if self.save_ and self.save_path is not None:
            cv2.imwrite(str(save_path), self.canvas)

            return save_path

    def move_cursor(self, direction="right", amount=(0, 0)):

        self._move_cursor(direction=direction, amount=amount)

    def _move_cursor(self, direction="right", amount=0):

        if direction == "right":
            self.cursor = (self.cursor[0], self.cursor[1] + amount[1])
        elif direction == "left":
            self.cursor = (self.cursor[0], self.cursor[1] - amount[1])
        elif direction == "up":
            self.cursor = (self.cursor[0] - amount[0], self.cursor[1])
        elif direction == "down":
            self.cursor = (self.cursor[0] + amount[0], self.cursor[1])
        elif direction == "center":
            self.cursor = (self.cursor[0] + amount[0], self.cursor[1] + amount[1])
        elif direction == "left-up":
            self.cursor = (self.cursor[0] - amount[0], self.cursor[1] - amount[1])
        elif direction == "right-up":
            self.cursor = (self.cursor[0] - amount[0], self.cursor[1] + amount[1])
        elif direction == "left-down":
            self.cursor = (self.cursor[0] + amount[0], self.cursor[1] - amount[1])
        elif direction == "right-down":
            self.cursor = (self.cursor[0] + amount[0], self.cursor[1] + amount[1])
        elif direction == "MASTER-CENTER":
            self.cursor = (self.height // 2, self.width // 2)
        elif direction == "MASTER-LEFT":
            self.cursor = (self.height // 2, 0)
        elif direction == "MASTER-RIGHT":
            self.cursor = (self.height // 2, self.width)
        elif direction == "MASTER-UP":
            self.cursor = (0, self.width // 2)
        elif direction == "MASTER-DOWN":
            self.cursor = (self.height, self.width // 2)
        elif direction == "MASTER-LEFT-UP":
            self.cursor = (0, 0)
        elif direction == "MASTER-RIGHT-UP":
            self.cursor = (0, self.width)
        elif direction == "MASTER-LEFT-DOWN":
            self.cursor = (self.height, 0)
        elif direction == "MASTER-RIGHT-DOWN":
            self.cursor = (self.height, self.width)
        elif direction == "point":
            self.cursor = (amount[0], amount[1])
        elif direction == "none":
            pass
        else:
            raise ValueError("Invalid direction")

    def get_cursor(self):
        return self.cursor

    def set_default_config(self):
        self.config = {
            "width": 800,
            "height": 600,
            "channel": 3,
            "background_color": (0, 0, 0),
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "font_scale": 1,
            "font_color": COLORS.WHITE,
            "font_thickness": 2,
            "cursor": (0, 0),
            "name": "canvas",
            "show": True,
            "save": False,
            "create_date_time_path": True,
            "save_path": "./",
        }

    def update_config(self, config):
        self.config.update(config)
