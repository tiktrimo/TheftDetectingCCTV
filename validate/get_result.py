from os import listdir, makedirs, path
from os.path import isfile, join
import json 


def isSteal(result, acc, thr):
    total_acc = 0
    temp_acc = 0
    isTotalSteal = False
    isTempSteal = False
    isInit = True
    for value in result:
        pred_none, pred_steal = value
        
        if isInit:
            if pred_steal == 1.0: continue
            else: isInit = False

        if pred_steal > thr:
            total_acc += 1
            temp_acc += 1
            if temp_acc > acc:
                isTempSteal = True
        else:
            temp_acc = 0
    if total_acc > acc:
        isTotalSteal = True
            
    return isTotalSteal, isTempSteal

def read_result_path(result_path):
    result_file = open(result_path, "r")
    lines = result_file.readlines()
    result = [list(map(lambda x : float(x), line.split(" "))) for line in lines]
    result_file.close()
    
    return result

def main():
    source_path = "/home/hsm/Python/MMACTION2/validate/results/"
    result_paths = [join(source_path, f) for f in listdir(source_path) if isfile(join(source_path, f)) and 'txt' in f]
    
    totalDict = {}
    tempDict = {}
    for thr in range(80, 90):
        for acc in range(1, 60):
            for result_path in result_paths:
                result = read_result_path(result_path)
                sequence_length = int(result_path.split("/")[-1].split("_")[0][1:])
                result_type = result_path.split("/")[-1].split("_")[-1].split(".")[0]
                
                isTotalSteal, isTempSteal = isSteal(result, acc, thr / 100)  
                if sequence_length not in totalDict: totalDict[sequence_length] = {}
                if sequence_length not in tempDict: tempDict[sequence_length] = {}
                if thr not in totalDict[sequence_length]: totalDict[sequence_length][thr] = {}
                if thr not in tempDict[sequence_length]: tempDict[sequence_length][thr] = {}
                if acc not in totalDict[sequence_length][thr]: totalDict[sequence_length][thr][acc] = {'TP' : 0, "FN" : 0, "FP" : 0, "TN" : 0}
                if acc not in tempDict[sequence_length][thr]: tempDict[sequence_length][thr][acc] = {'TP' : 0, "FN" : 0, "FP" : 0, "TN" : 0}
                
                if result_type == "STEAL":
                    if isTotalSteal: totalDict[sequence_length][thr][acc]["TP"] += 1
                    else: totalDict[sequence_length][thr][acc]["FN"] += 1
                    if isTempSteal: tempDict[sequence_length][thr][acc]["TP"] += 1
                    else: tempDict[sequence_length][thr][acc]["FN"] += 1
                else:
                    if isTotalSteal: totalDict[sequence_length][thr][acc]["FP"] += 1
                    else: totalDict[sequence_length][thr][acc]["TN"] += 1
                    if isTempSteal: tempDict[sequence_length][thr][acc]["FP"] += 1
                    else: tempDict[sequence_length][thr][acc]["TN"] += 1
                    
    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_temp_dict.json", "w")
    result_file.write(json.dumps(tempDict, sort_keys=True))
    result_file.close()
    
    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_total_dict.json", "w")
    result_file.write(json.dumps(totalDict, sort_keys=True))
    result_file.close()

    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_temp_dict.json", "r")
    tempDict = json.load(result_file)
    result_file.close()
    
    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_total_dict.json", "r")
    totalDict = json.load(result_file)
    result_file.close()
                
                    
    totalResult = {'recall' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0},
                   'precision' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0}, 
                   'f1' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0}}
    for sequence_length in totalDict:
        for thr in totalDict[sequence_length]:
            for acc in totalDict[sequence_length][thr]:
                curr_dict = totalDict[sequence_length][thr][acc]
                recall = curr_dict["TP"] / (curr_dict["TP"] + curr_dict["FN"])
                precision = 0
                if curr_dict["TP"] + curr_dict["FP"] != 0.0:
                    precision = curr_dict["TP"] / (curr_dict["TP"] + curr_dict["FP"])
                f1 = 0
                if precision + recall != 0:
                    f1 = 2 * ((precision * recall) / (precision + recall))
                accuracy = (curr_dict["TP"] + curr_dict["TN"]) / (curr_dict["TP"] + curr_dict["FN"] + curr_dict["FP"] + curr_dict["TN"])
                if recall > totalResult['recall']['value']:
                    totalResult['recall']['sequence_length'] = sequence_length
                    totalResult['recall']['thr'] = thr
                    totalResult['recall']['acc'] = acc
                    totalResult['recall']['value'] = recall
                    totalResult['recall']['precision'] = precision
                    totalResult['recall']['f1'] = f1
                    totalResult['recall']['recall'] = recall
                    totalResult['recall']['accuracy'] = accuracy
                if precision > totalResult['precision']['value']:
                    totalResult['precision']['sequence_length'] = sequence_length
                    totalResult['precision']['thr'] = thr
                    totalResult['precision']['acc'] = acc
                    totalResult['precision']['value'] = precision
                    totalResult['precision']['precision'] = precision
                    totalResult['precision']['f1'] = f1
                    totalResult['precision']['recall'] = recall
                    totalResult['precision']['accuracy'] = accuracy
                if f1 > totalResult['f1']['value']:
                    totalResult['f1']['sequence_length'] = sequence_length
                    totalResult['f1']['thr'] = thr
                    totalResult['f1']['acc'] = acc
                    totalResult['f1']['value'] = f1
                    totalResult['f1']['precision'] = precision
                    totalResult['f1']['f1'] = f1
                    totalResult['f1']['recall'] = recall
                    totalResult['f1']['accuracy'] = accuracy
                    
    tempResult = {'recall' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0},
                   'precision' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0}, 
                   'f1' : {'value' : 0, 'sequence_length' : 0, 'thr' : 0, 'acc' : 0}}
    for sequence_length in tempDict:
        for thr in tempDict[sequence_length]:
            for acc in tempDict[sequence_length][thr]:
                curr_dict = tempDict[sequence_length][thr][acc]
                recall = curr_dict["TP"] / (curr_dict["TP"] + curr_dict["FN"])
                precision = 0
                if curr_dict["TP"] + curr_dict["FP"] != 0.0:
                    precision = curr_dict["TP"] / (curr_dict["TP"] + curr_dict["FP"])
                f1 = 0
                if precision + recall != 0:
                    f1 = 2 * ((precision * recall) / (precision + recall))
                accuracy = (curr_dict["TP"] + curr_dict["TN"]) / (curr_dict["TP"] + curr_dict["FN"] + curr_dict["FP"] + curr_dict["TN"])
                if recall > tempResult['recall']['value']:
                    tempResult['recall']['sequence_length'] = sequence_length
                    tempResult['recall']['thr'] = thr
                    tempResult['recall']['acc'] = acc
                    tempResult['recall']['value'] = recall
                    tempResult['recall']['precision'] = precision
                    tempResult['recall']['f1'] = f1
                    tempResult['recall']['recall'] = recall
                    tempResult['recall']['accuracy'] = accuracy
                if precision > tempResult['precision']['value']:
                    tempResult['precision']['sequence_length'] = sequence_length
                    tempResult['precision']['thr'] = thr
                    tempResult['precision']['acc'] = acc
                    tempResult['precision']['value'] = precision
                    tempResult['precision']['precision'] = precision
                    tempResult['precision']['f1'] = f1
                    tempResult['precision']['recall'] = recall
                    tempResult['precision']['accuracy'] = accuracy
                if f1 > tempResult['f1']['value']:
                    tempResult['f1']['sequence_length'] = sequence_length
                    tempResult['f1']['thr'] = thr
                    tempResult['f1']['acc'] = acc
                    tempResult['f1']['value'] = f1
                    tempResult['f1']['precision'] = precision
                    tempResult['f1']['f1'] = f1
                    tempResult['f1']['recall'] = recall
                    tempResult['f1']['accuracy'] = accuracy
    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_temp.json", "w")
    result_file.write(json.dumps(tempResult, indent=4))
    result_file.close()
    
    result_file = open("/home/hsm/Python/MMACTION2/validate/results/results_total.json", "w")
    result_file.write(json.dumps(totalResult, indent=4))
    result_file.close()
        
    
    return



if __name__ == '__main__':
    main()