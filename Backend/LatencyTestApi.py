import flask
from flask import Blueprint, request

from LatencyInferenceService import LatencyInferenceService


def create_latency_blueprint(classificator):
    """Expose latency-only endpoints while leaving the normal classification API untouched."""
    blueprint = Blueprint("latency_test", __name__)
    latency_service = LatencyInferenceService(classificator)

    def _get_required_content(required_fields):
        content = request.get_json(silent=True)

        if not content:
            raise ValueError("A JSON request body is required.")

        missing_fields = [
            field for field in required_fields
            if field not in content
        ]

        if missing_fields:
            raise ValueError(
                f"Missing request field(s): {', '.join(missing_fields)}")

        return content

    @blueprint.route("/AiService/LatencyExecutionUnit", methods=["POST"])
    def latency_execution_unit_requested():
        try:
            content = _get_required_content(("ExecutionUnit",))

            result = latency_service.configure_execution_unit(
                content["ExecutionUnit"])
        except ValueError as exception:
            return flask.Response(str(exception), status=400)

        return flask.jsonify(result)

    @blueprint.route("/AiService/LatencyPrepareImages", methods=["POST"])
    def latency_prepare_images_requested():
        try:
            content = _get_required_content(
                ("PathFrontal", "PathLateral"))

            latency_service.prepare_images(
                content["PathFrontal"],
                content["PathLateral"])
        except ValueError as exception:
            return flask.Response(str(exception), status=400)

        # The latency UI does not need the large preview-image JSON returned by
        # the normal LoadImages endpoint.
        return flask.Response(status=204)

    @blueprint.route("/AiService/LatencyReleaseImages", methods=["POST"])
    def latency_release_images_requested():
        try:
            content = _get_required_content(
                ("PathFrontal", "PathLateral"))

            latency_service.release_images(
                content["PathFrontal"],
                content["PathLateral"])
        except ValueError as exception:
            return flask.Response(str(exception), status=400)

        return flask.Response(status=204)

    @blueprint.route("/AiService/LatencyClassification", methods=["POST"])
    def latency_classification_requested():
        try:
            content = _get_required_content(
                (
                    "ModelFrontal",
                    "ModelLateral",
                    "PathFrontal",
                    "PathLateral"
                ))

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
