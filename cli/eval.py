from cli import utils_init

def main():
    config, logger = utils_init.experiment_initialization()

    if 'seed' in config["evaluator"]["params"]:
        random_seed = config["trainer"]["params"].pop('seed')
        logger.info(f'random seed: {random_seed}')
        utils_init.set_seed(random_seed)

    evaluator = utils_init.build_pipeline(config)
    print(evaluator)

    
    evaluator.eval()



if __name__ == "__main__":


    main()