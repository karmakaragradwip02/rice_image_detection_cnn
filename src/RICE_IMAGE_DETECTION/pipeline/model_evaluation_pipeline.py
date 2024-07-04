from RICE_IMAGE_DETECTION.config.configuration import ConfigureationManager
from RICE_IMAGE_DETECTION.components.model_evaluation import ModelEvaluation 
from RICE_IMAGE_DETECTION import logger

STAGE_NAME = "MODEL EVALUATION STAGE"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigureationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        model = evaluation.load_model()
        val_set = evaluation.val_set()
        evaluation.plot()
        evaluation.log_into_mlflow(model, val_set)
    
if __name__ == '__main__':
    try: 
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e