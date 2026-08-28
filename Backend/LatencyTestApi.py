import flask
from flask import Blueprint, request

from LatencyInferenceService import LatencyInferenceService


def create_latency_blueprint(classificator):
    """Expose the latency endpoint while leaving the normal classification API untouched."""
    blueprint = Blueprint("latency_test", __name__)
    latency_service = LatencyInferenceService(classificator)

    @blueprint.route("/AiService/LatencyClassification", methods=["POST"])
    def latency_classification_requested():
        content = request.get_json(silent=True)

        if not content:
            return flask.Response(
                "A JSON request body is required.",
                status=400)

        required_fields = (
            "ModelFrontal",
            "ModelLateral",
            "PathFrontal",
            "PathLateral")

        missing_fields = [
            field for field in required_fields
            if field not in content
        ]

        if missing_fields:
            return flask.Response(
                f"Missing request field(s): {', '.join(missing_fields)}",
                status=400)

        try:
            result = latency_service.classify(
                content["ModelFrontal"],
                content["ModelLateral"],
                content["PathFrontal"],
                content["PathLateral"])
        except ValueError as exception:
            return flask.Response(str(exception), status=400)

        print(
            "Latency classification done. "
            f"Results: {result['OutputFrontal']} | {result['OutputLateral']}. "
            f"Inference timing: frontal="
            f"{result['FrontalInferenceMilliseconds']:.3f} ms, "
            f"lateral={result['LateralInferenceMilliseconds']:.3f} ms, "
            f"total={result['InferenceMilliseconds']:.3f} ms, "
            f"execution={result['ExecutionProvider']}, "
            f"device={result['TimingDevice']}, "
            f"method={result['TimingMethod']}")

        return flask.jsonify(result)

    return blueprint
