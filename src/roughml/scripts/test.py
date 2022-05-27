txt = 'dataset_1000_128_03_00_03_04_04_1.00.pt'
attrs = ["imgs", "dim", "rms", "skew", "kurt", "corrX", "corrY"] # attribute list to be added to title
char_index_list = []
char_index = 0
while char_index < len(txt):
    char_index = txt.find('_', char_index)
    if char_index == -1: # end of string or character not found
        del char_index_list[0] # ignore first underscore in title
        temp4 = 0
        for idx, char in enumerate(char_index_list):
            temp1 = txt[:char + temp4]
            temp2 = attrs[idx]
            temp3 = txt[char + temp4:]
            txt = txt[:char + temp4] + attrs[idx] + txt[char + temp4:]
            char_index_list = [x + (len(attrs[idx])+(char_index_list[idx+1]-char_index_list[idx])) for x in char_index_list] # update list with newly added string length
            temp4 = len(attrs[idx]) #+ (char_index_list[idx+1]-char_index_list[idx]) # update index with newly added string length
        break
    char_index_list.append(char_index)
    char_index += 1
