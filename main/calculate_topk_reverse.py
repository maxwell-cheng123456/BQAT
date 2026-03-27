from rdkit import Chem

def are_same_molecule(smiles1, smiles2):
    # 将SMILES字符串转换为分子对象
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
    except:
        return None

    if mol1 is None or mol2 is None:
        return None

    standard_smiles1 = Chem.MolToSmiles(mol1)
    standard_smiles2=Chem.MolToSmiles(mol2)

    if standard_smiles1==standard_smiles2:
        similarity = 1
    else:
        similarity=0
    return similarity

def SPLIT_SMILES(src):
    src=''.join(src.split(' '))
    reactant=src
    if reactant!=[]:
        reactant=reactant.split('.')
    return reactant
def is_not_in_predict(i,predict):
    for j in predict:
        similarity = are_same_molecule(i, j)
        # print(i, correct,similarity)
        if similarity == None:
            continue
        else:
            if similarity == 1:
                return 0
    return 1
def Compare_mols(predict,target):
    if len(predict)!=len(target):
        return 0
    for i in target:
        if is_not_in_predict(i,predict):
            return 0
    return 1
def are_equal(src,predict):
    try:

        r1=SPLIT_SMILES(src)
        r2 = SPLIT_SMILES(predict)
    except Exception as e:
        print(e)
        return 0
    if Compare_mols(r1,r2):
        return 1
    else:
        return 0
def calculateTopk(k,data_path,save_path):
    with open(data_path,'r')as f:
        srcs=f.readlines()

    with open(save_path,'r')as f1:
        preds=f1.readlines()
    correct=0
    all=0
    for i in range(len(srcs)):
        smiles1=srcs[i]
        all+=1
        for j in range(k):
            smiles2=preds[i*10+j]
            similarity = are_equal(smiles1, smiles2)
            # print(i, correct,similarity)
            if similarity == 0:
                continue
            else:
                if similarity == 1:
                    correct += 1
                    break
    return correct/all


if __name__=='__main__':
    experiment='S126'
    data_path = '../Data/canonical/src-test.txt'
    save_path = f'../experiment/S126_prediction1.txt'
    result=[]
    for k in range(10):
        result.append(f'top-{k+1}={calculateTopk(k=k+1,data_path=data_path,save_path=save_path)}')
    print(result)



