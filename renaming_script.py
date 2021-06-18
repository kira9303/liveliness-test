

#prepared = prepare()
import os
    
path = "D:/Emotion_detection/Song_test/Sad/"

new_path = "D:/Emotion_detection/Song_test/New_sad/"
    
def main(): 
  
    #import os

    for count, filename in enumerate(os.listdir(path)): 
        dst ="sad" + str(count) + ".mp3"
        src = path + filename 
        dst = new_path + dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 