from cli import utils_init

def main():
    config, logger = utils_init.experiment_initialization()

    '''Main training method'''
    if 'seed' in config["trainer"]["params"]:
        random_seed = config["trainer"]["params"].pop('seed')
        logger.info(f'random seed: {random_seed}')
        utils_init.set_seed(random_seed)

    trainer = utils_init.build_pipeline(config)
    print(trainer)

    
    trainer.train()



if __name__ == "__main__":


    main()