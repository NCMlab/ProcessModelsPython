# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
   
def main():
    print(len(sys.argv[1:]))
    if len(sys.argv[1:]) != 3:
        print("ERROR")
    else:
        print(sys.argv[1:][0])
        for arg in sys.argv[1:]:
            print(arg)
        
if __name__ == "__main__":
    main()
