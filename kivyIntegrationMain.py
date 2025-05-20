from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from SignToVoiceScreen import SignToVoiceScreen
import cv2
from SignToTextIntegrated import get_frame
#from AutoTTs import AutoTTs

Window.clearcolor = (1, 1, 1, 1)

class DetectionScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10

        self.image_widget = Image()
        self.add_widget(self.image_widget)

        self.label = Label(
            text='',
            font_size='20sp',
            size_hint=(1, 0.1),
            color=(0, 0, 0, 1)
        )
        self.add_widget(self.label)

        self.sign_to_voice_button = Button(
            text='Sign to Voice',
            size_hint=(1, 0.15),
            font_size='18sp',
            background_color=(0.1, 0.5, 0.2, 1),
            color=(1, 1, 1, 1),
            background_normal=''
        )
        self.sign_to_voice_button.bind(on_press=self.voice_pressed)
        self.add_widget(self.sign_to_voice_button)

        self.back_button = Button(
            text='Back',
            size_hint=(1, 0.1),
            font_size='16sp',
            background_color=(0.5, 0.1, 0.1, 1),
            color=(1, 1, 1, 1),
            background_normal=''
        )
        self.back_button.bind(on_press=self.go_back)
        self.add_widget(self.back_button)

        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS   

    def update(self, dt):
        frame, prediction = get_frame()
        if frame is not None:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]),
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture
            self.label.text = prediction

    def voice_pressed(self, instance):
        self.clear_widgets()
        self.add_widget(SignToVoiceScreen())

    def go_back(self, instance):
        self.clear_widgets()
        self.add_widget(OnboardingScreen())


class OnboardingScreen(BoxLayout):
    def __init__(self, **kwargs):
        super(OnboardingScreen, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 30
        self.spacing = 20

        self.add_widget(Label(
            text='Welcome to [b]GestuTalk[/b]',
            markup=True,
            font_size='24sp',
            color=(0, 0, 0, 1),
            size_hint=(1, 0.1),
            halign='center',
            valign='middle'
        ))

        self.image = Image(
            source='img.png',
            size_hint=(1, 0.4),
            allow_stretch=True
        )
        self.add_widget(self.image)

        self.continue_button = Button(
            text='Get Started',
            size_hint=(1, 0.15),
            font_size='18sp',
            background_color=(0.1, 0.2, 0.5, 1),
            color=(1, 1, 1, 1),
            background_normal=''
        )
        self.continue_button.bind(on_press=self.continue_pressed)
        self.add_widget(self.continue_button)

    def continue_pressed(self, instance):
        print("Continue button pressed - Switch to detection screen")
        self.clear_widgets()
        self.add_widget(DetectionScreen())


class GestuTalkApp(App):
    def build(self):
        return OnboardingScreen()


if __name__ == '__main__':
    GestuTalkApp().run()
