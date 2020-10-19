"""
Author: Jason Hayes
Description: The user inputs a word or phrase for the cipher to encrypt, and a 'key' to use for the encryption process.
  The key is a number between 0 and 26, and represents how many letters down the alphabet to move each character.  The
  Cipher will print the position in the alphabet of each character in the passphrase, its encryption, the position in
  the alphabet of each character in the encryption, and the decryption using the key to ensure encryption accuracy.

"""


passphrase = ''
key = 0
encryption = []
decryption = []

alphabet = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
            14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y',
            26: 'Z'}

def start_encryption():
    not_pass = True
    while not_pass:
        passphrase = str(input("Please enter the passphrase you would like to encrypt. The passphrase must be "
                               "all letters: "))
        if passphrase.isalpha():
            not_pass = False
            passphrase = passphrase.upper()
            print(passphrase)
        else:
            print("Please only enter letters")

    not_key = True
    while not_key:
        key = str(input("Please enter the key to use for encryption. Please enter a number between 0 and 26: "))
        if key.isnumeric():
            key = int(key)
            if key < 26 and key > 0:
                not_key = False
            else:
                print("Error. Please enter a number between 0 and 26")
        else:
            print("Please only enter numbers")

    encryptor(identify_number(passphrase), key)

    decryptor(identify_number(encryption), key)

def identify_number(passphrase):
    alphanum = []
    for char in passphrase:
        for k, v in alphabet.items():
            if char == v:
                alphanum.append(k)
    print(alphanum)
    return alphanum



def encryptor(alphanum, key):
    x = 1
    for num in alphanum:
        step = num + key
        if step > 26:
            step -= 26
        encryption.append(alphabet[step])
    print(encryption)

def decryptor(alphanum, key):
    for num in alphanum:
        step = num - key
        if step < 1:
            step += 26
        decryption.append(alphabet[step])
    print(decryption)

start_encryption()