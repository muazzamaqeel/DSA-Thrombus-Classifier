from functools import wraps
import flask
import torch
from flask import Blueprint, request, g
from LatencyInferenceService import LatencyInferenceService, LatencyCancelled
from LatencyDiagnostics import get_latency_info


def create_latency_blueprint(classificator):
    blueprint = Blueprint('latency_test', __name__)
    service = LatencyInferenceService(classificator)
    classificator.latency_service = service

    @blueprint.before_app_request
    def reserve_execution():
        if request.path in ('/AiService/LatencyInfo', '/AiService/LatencyCancel'):
            return None
        if not service._lock.acquire(blocking=False):
            return flask.Response('Backend busy. Wait for the active request or stop it.', status=409)
        g.latency_execution_reserved = True
        if not request.path.startswith('/AiService/Latency') and service._run_id:
            return flask.Response('Latency test is active. Stop/end it first.', status=409)

    @blueprint.teardown_app_request
    def release_execution(error):
        if getattr(g, 'latency_execution_reserved', False):
            g.latency_execution_reserved = False
            service._lock.release()

    def guarded(function):
        @wraps(function)
        def invoke():
            try:
                return function()
            except LatencyCancelled as error:
                return flask.Response(str(error), status=409)
            except ValueError as error:
                return flask.Response(str(error), status=400)
            except (MemoryError, torch.cuda.OutOfMemoryError) as error:
                # No automatic retry and no silent change of device/batch/precision.
                service.cancel(service._run_id)
                return flask.Response('Memory limit reached; test stopped. ' + str(error), status=507)
            except Exception as error:
                service.cancel(service._run_id)
                flask.current_app.logger.exception('Latency request failed')
                return flask.Response(str(error), status=500)
        return invoke

    def body(*required):
        content = request.get_json(silent=True)
        if not isinstance(content, dict):
            raise ValueError('A JSON object is required.')
        if any(name not in content for name in required):
            raise ValueError('Missing required request fields.')
        return content

    @blueprint.route('/AiService/LatencyInfo', methods=['GET'])
    @guarded
    def info():
        return flask.jsonify(get_latency_info())

    @blueprint.route('/AiService/LatencyExecutionMode', methods=['POST'])
    @guarded
    def execution_mode():
        content = body('Mode', 'ModelFolder', 'RunId')
        if service._run_id is not None:
            return flask.Response('A latency run is already active.', status=409)
        try:
            return flask.jsonify(service.configure_execution(content['Mode'], content['ModelFolder'],
                                 content.get('DeviceIndex', 0), content['RunId']))
        except Exception:
            service.end_run(content['RunId'])
            raise

    @blueprint.route('/AiService/LatencyCancel', methods=['POST'])
    @guarded
    def cancel():
        service.cancel(body('RunId')['RunId'])
        return flask.Response(status=204)

    @blueprint.route('/AiService/LatencyEnd', methods=['POST'])
    @guarded
    def end():
        service.end_run(body('RunId')['RunId'])
        return flask.Response(status=204)

    @blueprint.route('/AiService/LatencyPrepareImages', methods=['POST'])
    @guarded
    def prepare():
        content = body('PathFrontal', 'PathLateral', 'RunId')
        return flask.jsonify(service.prepare_images(content['PathFrontal'], content['PathLateral'], content['RunId']))

    @blueprint.route('/AiService/LatencyReleaseImages', methods=['POST'])
    @guarded
    def release():
        content = body('PathFrontal', 'PathLateral', 'RunId')
        service.release_images(content['PathFrontal'], content['PathLateral'], content['RunId'])
        return flask.Response(status=204)

    @blueprint.route('/AiService/LatencyClassification', methods=['POST'])
    @guarded
    def classify():
        content = body('ModelName', 'PathFrontal', 'PathLateral', 'RunId')
        return flask.jsonify(service.classify(content['ModelName'], content['PathFrontal'],
                                              content['PathLateral'], content['RunId']))

    return blueprint
