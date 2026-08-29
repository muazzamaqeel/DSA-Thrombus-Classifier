import flask
from flask import Blueprint, request

from LatencyInferenceService import LatencyInferenceService


def create_latency_blueprint(classificator):
    """Latency-only HTTP transport; normal application endpoints stay unchanged."""
    blueprint = Blueprint("latency_test", __name__)
    service = LatencyInferenceService(classificator)

    def body(*required):
        content = request.get_json(silent=True)
        if not content:
            raise ValueError("A JSON request body is required.")
        missing = [name for name in required if name not in content]
        if missing:
            raise ValueError(f"Missing request field(s): {', '.join(missing)}")
        return content

    @blueprint.route("/AiService/LatencyExecutionMode", methods=["POST"])
    def execution_mode():
        try:
            content = body("Mode", "ModelFolder")
            return flask.jsonify(service.configure_execution(
                content["Mode"], content["ModelFolder"]))
        except ValueError as error:
            return flask.Response(str(error), status=400)

    @blueprint.route("/AiService/LatencyPrepareImages", methods=["POST"])
    def prepare():
        try:
            content = body("PathFrontal", "PathLateral")
            service.prepare_images(content["PathFrontal"], content["PathLateral"])
            return flask.Response(status=204)
        except ValueError as error:
            return flask.Response(str(error), status=400)

    @blueprint.route("/AiService/LatencyReleaseImages", methods=["POST"])
    def release():
        try:
            content = body("PathFrontal", "PathLateral")
            service.release_images(content["PathFrontal"], content["PathLateral"])
            return flask.Response(status=204)
        except ValueError as error:
            return flask.Response(str(error), status=400)

    @blueprint.route("/AiService/LatencyClassification", methods=["POST"])
    def classify():
        try:
            content = body("ModelName", "PathFrontal", "PathLateral")
            return flask.jsonify(service.classify(
                content["ModelName"],
                content["PathFrontal"],
                content["PathLateral"]))
        except ValueError as error:
            return flask.Response(str(error), status=400)

    return blueprint
