
import os
import numpy as np
import pandas as pd
from clsmml import causal_direction


def  estemated_direction(extracted_dir_path):
    extracted_files = sorted(os.listdir(extracted_dir_path), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else 0)
    estimated_direction = []
    confidence=[] 
    t=[] 
    for index, file_name in enumerate(extracted_files):   
        file_path = os.path.join(extracted_dir_path, file_name)
        if os.path.isfile(file_path):
            x = []
            y = []
            if os.path.basename(file_path) != "pairs_gt.txt":
                
                with open(file_path, 'r') as file:
   
                    if file_name != "pairmeta.txt":
                    
                        first_line = file.readline().strip()
                        if first_line.startswith('"') :
                            for line in file:

                               if line.strip() :
                                   parts = line.strip().split(',')
                                   x.append(float(parts[1]))                                 
                                   y.append(float(parts[2]))
                                      
                        else:
                            for line in file:
                                
                                if line.strip():  # Skip empty lines
                                    parts = line.strip().split()
                                    x.append(float(parts[0]))
                                    y.append(float(parts[1]))
                        x = np.array(x)
                        y = np.array(y)
                        
                    else:
                        with open(file_path, 'r') as file:
                            
                            for line in file:
                                if line.strip():
                                    parts = line.split()
                      
                                    if parts[1] == '1':
                                        t.append(1)
                                    elif parts[1] =='2':
                                        t.append(0)
                    
            elif os.path.basename(file_path) == "pairs_gt.txt":
            
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.strip() :
                            parts = line.strip()
                            t.append(float(parts[0]))
                            print(parts[0])
                          
            if(len(x)!=0):
               estimated,conf= causal_direction(x, y)
               print(file_name,estimated)
               
               estimated_direction.append(estimated)
               confidence.append(conf)

    return estimated_direction,confidence,t


def  estemated_direction_2(extracted_dir_path):

    extracted_files = sorted(os.listdir(extracted_dir_path), key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else 0)
    estimated_direction = []
    confidence=[]
    t=[]
    estimated_direction = []
    confidence=[]
    for index, file_name in enumerate(extracted_files):

        file_path = os.path.join(extracted_dir_path, file_name)
        if os.path.isfile(file_path):
            x = []
            y = []
            with open(file_path, 'r') as file:
                  
                if 'pairs' in file_name:
                    df = pd.read_csv(file)
                    for index, row in df.iterrows():
                        x= [float(num) for num in row.iloc[1].split()]
                        y= [float(num) for num in row.iloc[2].split()]
                        x = np.array(x)
                        y = np.array(y)                           
                        estimated,conf = causal_direction(x, y)
                        print(index,estimated)
                        confidence.append(conf)                    
                        estimated_direction.append(estimated)
                elif 'targets' in file_name:
                      df = pd.read_csv(file)
                      t = df.iloc[:, 1].values
                      
    return estimated_direction,confidence,t


