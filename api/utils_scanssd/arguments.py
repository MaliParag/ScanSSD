import os

class Arguments:
    def __init__(self,trained_model='weights/AMATH512_e1GTDB.pth',visual_threshold=0.6,cuda=False,verbose = False, exp_name='SSD', model_type = 512, 
                 use_char_info=False,limit = -1, cfg = "hboxes512",batch_size = 16, num_workers = 4, kernel = [1,5], padding = [3,3],neg_mining = True,
                 stride = 0.1, window = 1200,root_folder = "..\..\files",stitching_algo='equal',algo_threshold=30,preprocess=False,postprocess=False,gen_math=True):
        
        self.trained_model = trained_model   #Modelo pré-treinado
        self.exp_name = exp_name # Nome do Experimento, usado para salvar os resultados
        self.root_folder = root_folder #pasta root onde estão armazenados os arquivos PDF, Imagem, resultados, etc
        self.save_folder = os.path.join(root_folder,self.exp_name,'results')       #Diretório para salvar os resultados
        self.visual_threshold = visual_threshold #Final confidence Threshold
        self.cuda = cuda #Usa ou não GPU
        self.dataset_root = os.path.join(self.root_folder,self.exp_name) #Local onde as imagens das páginas do PDF estão armazenadas
        self.test_data = os.path.join('data') #arquivo que relaciona linha a linha as imagens do PDF a serem processadas pelo ScanSSD
        self.verbose = verbose 
        self.model_type = model_type # Tipo do Modelo do ScanSSD SSD300 ou SSD512
        self.use_char_info = use_char_info # Se usa ou não a informação de caracter
        self.limit = limit # Se limita o número de exemplos, -1 para sem limite
        self.cfg = cfg # Tipo da Rede, se gtdb ou math_gtdb_512
        self.batch_size = batch_size # Batch size para treino ou teste
        self.num_workers = num_workers # Número de workers usado no carregamento dos dados
        self.kernel = kernel # Kernel para a camada de features
        self.padding = padding # Padding para a camada de features
        self.neg_mining = neg_mining # Se usa ou não hard negatibe mining com proporção 1:3
        self.log_dir = os.path.join(root_folder,self.exp_name,'logs') # Diretorio para salvamento dos Logs
        self.stride = stride # Stride para usar no processo de Scan da imagem
        self.window = window # Tamanhop da janela de sliding

        #Argumentos para o stitch
        self.data_file = os.path.join(self.root_folder,self.exp_name,'file_to_stitch') #arquivo que contém o nome do PDF a ser processado
        self.output_dir = os.path.join(self.save_folder,'stitch')
        self.math_dir = os.path.join(self.save_folder,self.exp_name)
        self.math_ext = '.csv'
        self.home_data = self.dataset_root #definir melhor essa variável
        self.home_eval = self.save_folder
        self.home_images = os.path.join(self.root_folder,self.dataset_root,'images')
        self.home_anno = os.path.join(root_folder,self.exp_name,'annotations')
        self.home_char = os.path.join(root_folder,self.exp_name,'annotations_char')
        self.stitching_algo = stitching_algo
        self.algo_threshold = algo_threshold
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.gen_math = gen_math

        #Argumentos para visualize
        self.img_dir = os.path.join(self.dataset_root,'images')
        self.output_dir_annot = os.path.join(self.root_folder,self.exp_name,'annotated')
        self.math_dir_annot = self.output_dir
        self.char_dir = None



