import tornado.ioloop
import tornado.web
import json

from service import TextClassificationApp
from service import TextClassificationService


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', '*')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        # self.set_header('Access-Control-Max-Age', 1000)

    def prepare(self):
        if self.request.method == "GET":
            self.json_args = None
            return

        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = self.request.body
            return
        self.json_args = None

    def options(self):
        pass


class MainHandler(BaseHandler):
    def get(self):
        self.write("<p>This is a demonstration of the nlp platform.<p>")
        self.write("<p>To use this demo, please send JSON data to /api/allennlp/ner through POST request.<p>")


class TextClassificationHandler(BaseHandler):
    def initialize(self, text_classifier: TextClassificationService):
        self.classifier = text_classifier

    def get(self):
        pass

    def post(self):
        if self.json_args:
            text = json.loads(self.json_args)
            predict = self.classifier.predict(text)

            self.set_header("Content-Type", "text/plain")
            self.write(predict)
            self.flush()
            self.finish()


def start_text_classification_service():
    app = TextClassificationApp()
    service = TextClassificationService(app)
    return service




def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/tcl", TextClassificationHandler, dict(text_classifier=start_text_classification_service()))
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8104)
    print("app started......")
    tornado.ioloop.IOLoop.current().start()
