import logging

from preprocessor import PreProcessor
from utils.config import load_patient_configs
import argparse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # load config file from first argument
    parser = argparse.ArgumentParser(description="Run preprocessing for patients.")
    parser.add_argument('-config_file', '-c', type=str, help='Path to the configuration file.')
    parser.add_argument('-device', '-d', type=str, default='cpu', help='Device to use for processing, e.g. cpu for CPU, cuda:0 for GPU 0,')
    parser.add_argument('-stage', '-s', type=int, nargs='+', default=None, help='Stages to run, e.g. -s 1 2 3')
    parser.add_argument('--overview', action='store_true', help='Generate an overview image without running the preprocessing pipeline.')
    parser.add_argument('--skip_recon', action='store_true', help='Skip rtk CBCT reconstruction.')
    parser.add_argument('--correct_contrast_media', action='store_true', help='Enable contrast media correction in stage 3 (requires PyTorch).')
    parser.add_argument('--cleanup', action='store_true', help='Delete generated files for the given stage instead of running it.')
    args = parser.parse_args()
    configs = load_patient_configs(args.config_file)

    for patient_id, config in configs.items():
        try:
            logger.info(f"--- Processing patient {patient_id} --- stage(s): {args.stage} ---")
            if args.correct_contrast_media:
                config.settings.correct_contrast_media = True
            processor = PreProcessor(patient_id, config, device=args.device, skip_recon=args.skip_recon)
            # delete files for stage 2 or 3
            for stage in args.stage:
                if args.cleanup:
                    if stage == 2:
                        processor.cleanup_s2()
                    elif stage == 3:
                        processor.cleanup_s3()

                elif stage == 1:
                    if args.overview:
                        processor.generate_overview_image()
                    else:
                        processor.run_stage1()

                elif stage == 2:
                    if args.overview:
                        processor.generate_overview_deformed()
                    else:
                        processor.run_stage2()

                elif stage == 3:
                    processor.run_stage3()
            logger.info(f"--- Successfully finished patient {patient_id} ---")

        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}", exc_info=True)

