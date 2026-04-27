import logging

import torch
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
    parser.add_argument('-stage', '-s', type=int, default=1, help='Preprocessing stage to run (1 or 2). ')
    parser.add_argument('--overview', action='store_true', help='Generate an overview image without running the preprocessing pipeline.')
    parser.add_argument('--skip_recon', action='store_true', help='Skip rtk CBCT reconstruction.')
    args = parser.parse_args()
    configs = load_patient_configs(args.config_file)

    for patient_id, config in configs.items():
        try:
            logger.info(f"--- Processing patient {patient_id} ---")
            processor = PreProcessor(patient_id, config, device=torch.device(args.device), skip_recon=args.skip_recon)
            if args.stage == 1:
                if args.overview:
                    processor.generate_overview_image()
                else:
                    processor.run_stage1()
            elif args.stage == 2:
                if args.overview:
                    processor.generate_overview_deformed()
                else:
                    processor.run_stage2()
            logger.info(f"--- Successfully finished patient {patient_id} ---")

        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}", exc_info=True)

