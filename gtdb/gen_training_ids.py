# Author: Parag Mali
# This script splits training data into 80% training and 20% validation

def gen_training_ids():

    dataset = [('Burstall77',24,[23,20,3,5,15]), ('BAMS_1998_123_143',21,[2,17,12,11]),
               ('AIF_1999_375_404',30,[16,17,18,9,24,15]), ('ASENS_1997_367_384',18,[11,17,7,6]),
               ('Brezis83',5,[5]), ('MA_1977_275_292',18,[4,2,5,7]), ('Borcherds86',4,[2]),
               ('BAMS_1971_1974_1',3,[1]), ('BAMS_1971_1974_2',4,[4]), ('MA_1999_175_196',22,[19,1,10,16]),
               ('JMS_1975_497_506',10,[6,3]), ('JMKU_1971_377_379',3,[3]), ('BAMS_1971_1974_3',4,[1]),
               ('AnnM_1970_550_569',20,[7,18,8,17]), ('AIF_1970_493_498',6,[1]), ('JMS_1975_281_288',8,[8,7]),
               ('TMJ_1990_163_193',32,[1,23,31,28,5,9]), ('TMJ_1973_317_331',16,[7,11,16]),
               ('MA_1970_26_38',13,[7,1,12]), ('InvM_1999_163_181',19,[18,10,12,2]),
               ('InvM_1970_121_134',14,[9,3,2]), ('BSMF_1970_165_192',28,[22,20,7,27,23,13]),
               ('ActaM_1998_283_305',23,[15,17,20,6,10]), ('ASENS_1970_273_284',12,[5,7]),
               ('TMJ_1973_333_338',6,[2]), ('Cline88',15,[14,9,11]),
               ('ActaM_1970_37_63',27,[21,19,8,27,23]), ('JMS_1975_289_293',5,[4]),
               ('BSMF_1998_245_271',27,[11,1,25,3,10]), ('Alford94',20,[10,11,5,13]),
               ('KJM_1999_17_36',20,[11,6,3,10]), ('JMKU_1971_181_194',14,[8,14,1]),
               ('Bergweiler83',37,[18,11,9,24,34,13,1]), ('Arkiv_1997_185_199',15,[9,6,2]),
               ('Arkiv_1971_141_163',23,[16,7,1,10,5]), ('JMKU_1971_373_375',3,[2])]

    print("Training dataset.....")
    for filename, page_count, val_pages in dataset:
        for i in range(1,page_count+1):
            if i not in val_pages:
                print(filename + "/" + str(i))

    print("Validation dataset.....")
    for filename, page_count, val_pages in dataset:
        for i in range(1, page_count + 1):
            if i in val_pages:
                print(filename + "/" + str(i))

if __name__ == '__main__':
    gen_training_ids()
