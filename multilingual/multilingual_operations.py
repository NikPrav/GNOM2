# from googletrans import Translator
# translator = Translator(service_urls=['translate.googleapis.com'])
#
#
# def detect_and_translate(text, target_lang='en'):
#     translated_text = translator.translate(text, dest=target_lang)
#     return translated_text
#
#
# from google_trans_new import google_translator
#
# translator = google_translator()
#
#
# def translate_google(txt):
#     translated_text = translator.translate(txt, lang_tgt='en')
#     return translated_text


from deep_translator import GoogleTranslator

dt_translator = GoogleTranslator(source='auto', target='de')


# dt_translator.translate('hola')

def translate_texts(texts):
    return dt_translator.translate_sentences(texts)
