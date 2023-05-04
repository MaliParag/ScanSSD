from api import scanssd

scanssd.test_gtdb(trained_model='weights\AMATH512_e1GTDB.pth',visual_threshold=0.6,cuda=False,verbose = False, exp_name='teste_api', model_type = 512, 
                 use_char_info=False,limit = -1, cfg = "hboxes512",batch_size = 16, num_workers = 4, kernel = [1,5], padding = [3,3],neg_mining = True,
                 stride = 0.1, window = 1200,root_folder = '../arquivos')

