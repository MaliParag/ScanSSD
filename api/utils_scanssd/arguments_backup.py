class Arguments:
    def __init__(self,trained_model='weights/AMATH512_e1GTDB.pth',save_folder='../results/',visual_threshold=0.6,cuda=False,dataset_root = '../results/images/',
                 test_data='../results/testing_data', verbose = False, exp_name='SSD', model_type = 512, use_char_info=False,limit = -1, cfg = "hboxes512",
                 batch_size = 16, num_workers = 4, kernel = [1,5], padding = [3,3],neg_mining = True,log_dir='../logs', stride = 0.1, window = 1200):
        
        self.trainded_model = trained_model   #Modelo pré-treinado
        self.save_folder = save_folder        #Diretório para salvar os resultados
        self.visual_threshold = visual_threshold #Final confidence Threshold
        self.cuda = cuda #Usa ou não GPU
        self.dataset_root = dataset_root #Local onde as imagens das páginas do PDF estão armazenadas
        self.test_data = test_data #arquivo que relaciona linha a linha as imagens do PDF a serem processadas pelo ScanSSD
        self.verbose = verbose
        self.exp_name = exp_name # Nome do Experimento, usado para salvar os resultados
        self.model_type = model_type # Tipo do Modelo do ScanSSD SSD300 ou SSD512
        self.use_char_info = use_char_info # Se usa ou não a informação de caracter
        self.limit = limit # Se limita o número de exemplos, -1 para sem limite
        self.cfg = cfg # Tipo da Rede, se gtdb ou math_gtdb_512
        self.batch_size = batch_size # Batch size para treino ou teste
        self.num_workers = num_workers # Número de workers usado no carregamento dos dados
        self.kernel = kernel # Kernel para a camada de features
        self.padding = padding # Padding para a camada de features
        self.neg_mining = neg_mining # Se usa ou não hard negatibe mining com proporção 1:3
        self.log_dir = log_dir # Diretorio para salvamento dos Logs
        self.stride = stride # Stride para usar no processo de Scan da imagem
        self.window = window # Tamanhop da janela de sliding
