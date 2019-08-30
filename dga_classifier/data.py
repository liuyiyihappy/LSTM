"""Generates data for train/test algorithms"""
# -*- coding:utf-8 -*-
# coding: utf-8
from datetime import datetime
#from StringIO import StringIO
import io
#from urllib import urlopen
from urllib.request import urlopen
#from zipfile import ZipFile
import zipfile

#import cPickle as pickle
import pickle as pickle
import os
import random
import tldextract



from dga_classifier.dga_generators import banjori, corebot, cryptolocker, dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda   #11种DGA算法

# Location of Alexa 1M
#ALEXA_1M = 'http://s3.amazonaws.com/alexa-static/top-1m.csv.zip'

ALEXA_1M = 'top-1m.csv.zip'

# Our ourput file containg all the training data
DATA_FILE = 'traindata.pkl'

# class StrToBytes:
#     def __init__(self, fileobj):
#         self.fileobj = fileobj
#     def read(self, size):
#         return self.fileobj.read(size).encode()
#     def readline(self, size=-1):
#         return self.fileobj.readline(size).encode()

def get_alexa(num, address=ALEXA_1M, filename='top-1m.csv'):   # 提取Alexa数据
    """Grabs Alexa 1M"""
    # url = urlopen(address)
    # zipfile = ZipFile(io.StringIO(url.read()))
    #zipfile = ZipFile(io.StringIO(url.read().decode('charmap')))

    zf = zipfile.ZipFile(address, 'r')
    #print(zipfile.read(filename))


    for x in zf.read(filename).decode().split()[:num]:    #返回分割后的字符串列表
        alexa_content = tldextract.extract(x.split(',')[1]).domain
        #print(alexa_content)

    return alexa_content    #返回string类型

    #return [tldextract.extract(x.split(',')[1]).domain for x in zipfile.read(filename).decode().split()[:num]] #tldextract准确地从URL的域名和子域名分离通用顶级域名或国家顶级域名

def gen_malicious(num_per_dga=10000):
    """Generates num_per_dga of each DGA"""
    domains = []
    labels = []

    # We use some arbitrary seeds to create domains with banjori
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = max(1, num_per_dga // len(banjori_seeds))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, banjori_seed)
        labels += ['banjori']*segs_size

    domains += corebot.generate_domains(num_per_dga)
    labels += ['corebot']*num_per_dga

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, num_per_dga // len(crypto_lengths))
    for crypto_length in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker']*segs_size

    domains += dircrypt.generate_domains(num_per_dga)
    labels += ['dircrypt']*num_per_dga

    # generate kraken and divide between configs
    kraken_to_gen = max(1, num_per_dga // 2)
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
    labels += ['kraken']*kraken_to_gen
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
    labels += ['kraken']*kraken_to_gen

    # generate locky and divide between configs
    locky_gen = max(1, num_per_dga // 11)
    for i in range(1, 12):
        domains += lockyv2.generate_domains(locky_gen, config=i)
        labels += ['locky']*locky_gen

    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa']*num_per_dga

    # Generate qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[])
    labels += ['qakbot']*num_per_dga

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = max(1, num_per_dga // len(ramdo_lengths))
    for rammdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        labels += ['ramdo']*segs_size

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
    labels += ['ramnit']*num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, num_per_dga // len(simda_lengths))
    for simda_length in range(len(simda_lengths)):
        domains += simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2**32))
        labels += ['simda']*segs_size


    return domains, labels  #109935


def gen_data(force=True):  #写入traindata.pkl
    """Grab all data for train/test and save

    force:If true overwrite, else skip if file
          already exists
    """
    if force or (not os.path.isfile(DATA_FILE)):
        domains, labels = gen_malicious(10000)


        # Get equal number of benign/malicious
        m_domains = len(domains)
        domains += get_alexa(len(domains))  #109935

        #labels += ['benign']*len(domains)
        labels += ['benign'] * m_domains    #给alexa数据打上benign标签


        #pickle.dump(zip(labels, domains), open(DATA_FILE, 'w'))
        #pickle.dump(zip(labels, domains), open(DATA_FILE, 'w').decode('charmap'))
        pickle.dump(list(zip(labels, domains)), open(DATA_FILE, 'wb'))  #序列化对象，将对象obj保存到文件file中去




def get_data(force=True):
    """Returns data and labels"""
    gen_data(force)

    # with open(DATA_FILE,'r') as data_file:
    #
    #     response = pickle.load(StrToBytes(data_file))  #反序列化对象，将文件中的数据解析为一个python对象
    #
    # return response

    with open(DATA_FILE,'rb') as data_file:

        response = pickle.load(data_file)  #反序列化对象，将文件中的数据解析为一个python对象


    return response

    #return pickle.load(open(DATA_FILE))
