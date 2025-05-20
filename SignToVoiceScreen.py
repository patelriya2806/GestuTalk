import cv2
import pyttsx3
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.app import App

from SignToTextIntegrated import get_frame
#from kivyIntegrationMain import OnboardingScreen  # Import if needed for navigation back

#cd /home/riya2806/GestuTalk
#sudo buildozer android debug
#buildozer android debug


engine = pyttsx3.init()

class SignToVoiceScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10

        self.image_widget = Image()
        self.add_widget(self.image_widget)

        self.speak_button = Button(
            text='🔊 Speak Prediction',
            size_hint=(1, 0.1),
            font_size='16sp',
            background_color=(0.2, 0.4, 0.6, 1),
            color=(1, 1, 1, 1),
            background_normal=''
        )
        self.speak_button.bind(on_press=self.speak_prediction)
        self.add_widget(self.speak_button)

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

        self.prediction = ""
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        frame, prediction = get_frame()
        if frame is not None:
            self.prediction = prediction
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

    def speak_prediction(self, instance):
        if self.prediction:
            clean_text = self.prediction.split("(")[0].strip()
            engine.say(clean_text)
            engine.runAndWait()

    def go_back(self, instance):
        app = App.get_running_app()
        app.root.clear_widgets()

        # Recreate onboarding layout here instead of importing
        from kivyIntegrationMain import OnboardingScreen  # ONLY import here
        app.root.add_widget(OnboardingScreen())

