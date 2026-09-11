import flask
import torch
from flask import Blueprint, request

from LatencyInferenceService import LatencyInferenceService
from LatencyDiagnostics import get_latency_info


def create_latency_blueprint(classificator):
    """Latency-only HTTP transport; normal application endpoints stay unchanged."""
    blueprint = Blueprint("latency_test", __name__)
    service = LatencyInferenceService(classificator)

    @blueprint.route("/AiService/LatencyInfo", methods=["GET"])
    def info():
        return flask.jsonify(get_latency_info())

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
                content["Mode"], content["ModelFolder"], content.get("DeviceIndex", 0)))
        except ValueError as error:
            return flask.Response(str(error), status=400)
        except torch.cuda.OutOfMemoryError:
            return flask.Response("The selected GPU cannot fit one model and this input. GPU execution stopped; no CPU fallback was used.", status=500)
        except RuntimeError as error:
            return flask.Response(str(error), status=500)

    @blueprint.route("/AiService/LatencyPrepareImages", methods=["POST"])
    def prepare():
        try:
            content = body("PathFrontal", "PathLateral", "RunId")
            return flask.jsonify(service.prepare_images(content["PathFrontal"], content["PathLateral"], content["RunId"]))
        except ValueError as error:
            return flask.Response(str(error), status=400)
        except torch.cuda.OutOfMemoryError:
            return flask.Response("The selected GPU cannot fit one model and this input. GPU execution stopped; no CPU fallback was used.", status=500)
        except RuntimeError as error:
            return flask.Response(str(error), status=500)

    @blueprint.route("/AiService/LatencyReleaseImages", methods=["POST"])
    def release():
        try:
            content = body("PathFrontal", "PathLateral", "RunId")
            service.release_images(content["PathFrontal"], content["PathLateral"], content["RunId"])
            return flask.Response(status=204)
        except ValueError as error:
            return flask.Response(str(error), status=400)
        except torch.cuda.OutOfMemoryError:
            return flask.Response("The selected GPU cannot fit one model and this input. GPU execution stopped; no CPU fallback was used.", status=500)
        except RuntimeError as error:
            return flask.Response(str(error), status=500)

    @blueprint.route("/AiService/LatencyClassification", methods=["POST"])
    def classify():
        try:
            content = body("ModelName", "PathFrontal", "PathLateral", "RunId")
            return flask.jsonify(service.classify(
                content["ModelName"],
                content["PathFrontal"],
                content["PathLateral"], content["RunId"]))
        except ValueError as error:
            return flask.Response(str(error), status=400)
        except torch.cuda.OutOfMemoryError:
            return flask.Response("The selected GPU cannot fit one model and this input. GPU execution stopped; no CPU fallback was used.", status=500)
        except RuntimeError as error:
            return flask.Response(str(error), status=500)

    return blueprint
