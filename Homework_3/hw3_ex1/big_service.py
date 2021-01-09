import base64
import cherrypy
import tensorflow as tf
import json
import numpy as np


class InferenceService(object):
    exposed = True

    def __init__(self, model='./Group2_kws_c.tflite'):
        self.model = model
        if self.model is not None: 
            self.interpreter = tf.lite.Interpreter(model_path=model)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

        labels_file = open("../labels.txt", "r")
        LABELS = labels_file.read()
        labels_file.close()
        self.labels = np.array(LABELS.split(" "))
        
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40, 321, 16000, 20, 4000)

    def preprocess(self, audio_bytes):
        # Decode and normalize
        audio, _ = tf.audio.decode_wav(audio_bytes)
        audio = tf.squeeze(audio, axis=1)

        # STFT
        stft = tf.signal.stft(audio, frame_length=640, frame_step=320, fft_length=640)
        spectrogram = tf.abs(stft)

        # MFCC
        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :10]

        # Add channel dimension
        mfccs = tf.expand_dims(mfccs, -1)
        # Add batch dimension
        mfccs = tf.expand_dims(mfccs, 0)

        return mfccs

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        input_body = cherrypy.request.body.read()
        input_body = json.loads(input_body)
        events = input_body['e']

        audio_string = None
        for event in events:
            audio_string = event['vd']
        if audio_string is None:
            raise cherrypy.HTTPError(400, 'no audio event')

        audio_bytes = base64.b64decode(audio_string)
        mfccs = self.preprocess(audio_bytes)

        model = self.model
        if model is None:
            raise cherrypy.HTTPError(400, 'no valid model')
        
        self.interpreter.set_tensor(input_details[0]['index'], mfcss)
        #start_inf = time.time()
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(output_details[0]['index'])
        #probs = tf.nn.softmax(logits)
        #prob = tf.reduce_max(probs).numpy() * 100
        label_idx = tf.argmax(logits, 1).numpy()[0]
        label = self.labels[label_idx]

        output_body = {
                        'label': label,
                        'probability': prob}
        output_body = json.dumps(output_body)

        return output_body

    def DELETE(self, *path, **query):
        pass

if __name__ == '__main__':
    conf = {'/': { 
                'request.dispatch': cherrypy.dispatch.MethodDispatcher()
                }} 
    cherrypy.tree.mount(InferenceService(), '/', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block() 