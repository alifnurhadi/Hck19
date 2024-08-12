import string
import random
import sys
import re
"""
Format penulisan syntax function '.join()'
separator_string.join(iterable) :

separator_string:

This is the string on which the .join() method is called.
It becomes the separator between each element in the final joined string.
It can be any string, including an empty string '', a single character like ',', or even a multi-character string like ', ' or ' and '.
"""

iterator= string.ascii_letters + string.digits
complex = iterator + '@$!_'

def PassGen(length:int) -> int:
    pw  = ''.join(random.choice(complex) for _ in range(length))
    return pw

def main():

    if len(sys.argv) <= 1:
        print(''' ur argument cannot be accept 
        \n but here's a 8 digit password for u (if want to get longer password put number in command-line): ''')
        print(PassGen(8))

    else: 
        args = sys.argv[1]

        if re.match(r'\d+',args):
            try:
                if int(args) > 7:
                    arg = int(args)
                    print(PassGen(arg))
                else:
                    print(''' ur argument cannot be accept 
                    \n but here's a 8 digit password for u (if want to get longer password put number in command-line): ''')
                    print(PassGen(8))

            except ValueError:
                    print("put number greater or equal to 8")
                    sys.exit(1)
                
        else:
            raise ValueError ('ur argument to small or not acceptable, please put number')



    """if len(sys.argv) <= 1:
        print(''' ur argument cannot be accept 
        \n but here's a 8 digit password for u (if want to get longer password put number in command-line): ''')
        print(PassGen(8))
    else:
        args = sys.argv[1]

        re_num = re.search(r'\d+',args)
        # re_word = re.search(r'^\w+$',args)

        # if len(str(re_word))>= 1:
        #     print("sorry cannot accept the given argument.")
        #     pass
        if len(str(re_num))> 0:
            if int(sys.argv[1])>7 :
                try:
                    arg = int(args)
                    print(PassGen(arg))

                except ValueError:
                    print("put number greater or equal to 8")
                    sys.exit(1)
            else:
                print(PassGen(8))
        else :
            raise ValueError ('ur argument to small or not acceptable, please put number')"""
    

    
if __name__=="__main__":
    try:
        main()
    except ValueError:
        print(''' ur argument cannot be accept 
                \n but here's a 8 digit password for u (if want to get longer password put number in command-line): ''')
        print(PassGen(8))

'''
PassGen(24)

cara pemanggilan printnya bisa dimodif kayaknya pake sys_arg function

'''
