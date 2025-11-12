from gtts import gTTS

text = "down"
lang = "en"

tts = gTTS(text=text, lang=lang)

tts.save("../resources/sounds/down.mp3")
