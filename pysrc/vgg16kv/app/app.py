from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty, ListProperty, NumericProperty


class StyleTransferImageRow(Widget):
    images = ListProperty([])


class StyleTransferImage(Widget):
    image_file = StringProperty("")
    size_x = NumericProperty(100)
    size_y = NumericProperty(100)


class StyleTransferBase(Widget):
    pass


class StyleTransferApp(App):
    def build(self):
        return StyleTransferBase()
