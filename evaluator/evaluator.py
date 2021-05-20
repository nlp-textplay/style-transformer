from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import fasttext
import pkg_resources
import kenlm
import math
import torch
from sim_models import WordAveraging
from sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm


class Evaluator(object):

    def __init__(self):
        resource_package = __name__

        yelp_acc_path = 'acc_style.bin'
        yelp_ppl_path = 'ppl_style.binary'
        yelp_ref0_path = 'yelp.refs.0'
        yelp_ref1_path = 'yelp.refs.1'
        sim_path = "sim.pt"
        sp_path = "sim.sp.30k.model"

        
        yelp_acc_file = pkg_resources.resource_stream(resource_package, yelp_acc_path)
        yelp_ppl_file = pkg_resources.resource_stream(resource_package, yelp_ppl_path)
        yelp_ref0_file = pkg_resources.resource_stream(resource_package, yelp_ref0_path)
        yelp_ref1_file = pkg_resources.resource_stream(resource_package, yelp_ref1_path)
        sim_file = pkg_resources.resource_stream(resource_package, sim_path)

        
        self.yelp_ref = []
        with open(yelp_ref0_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())
        with open(yelp_ref1_file.name, 'r') as fin:
            self.yelp_ref.append(fin.readlines())
        self.classifier_yelp = fasttext.load_model(yelp_acc_file.name)
        self.yelp_ppl_model = kenlm.Model(yelp_ppl_file.name)
        
        sim = torch.load(sim_file.name)
        state_dict = sim['state_dict']
        vocab_words = sim['vocab_words']
        args = sim['args']
        # turn off gpu
        self.sim = WordAveraging(args, vocab_words)
        self.sim.load_state_dict(state_dict, strict=True)
        sp = spm.SentencePieceProcessor()
        sp.Load(sp_path)
        self.sim.eval()
        self.tok = TreebankWordTokenizer()
        
    def yelp_style_check(self, text_transfered, style_origin):
        text_transfered = ' '.join(word_tokenize(text_transfered.lower().strip()))
        if text_transfered == '':
            return False
        label = self.classifier_yelp.predict([text_transfered])
        style_transfered = label[0][0] == '__label__positive'
        return (style_transfered != style_origin)

    def yelp_acc_b(self, texts, styles_origin):
        assert len(texts) == len(styles_origin), 'Size of inputs does not match!'
        count = 0
        for text, style in zip(texts, styles_origin):
            if self.yelp_style_check(text, style):
                count += 1
        return count / len(texts)

    def yelp_acc_0(self, texts):
        styles_origin = [0] * len(texts)
        return self.yelp_acc_b(texts, styles_origin)

    def yelp_acc_1(self, texts):
        styles_origin = [1] * len(texts)
        return self.yelp_acc_b(texts, styles_origin)

    def nltk_bleu(self, texts_origin, text_transfered):
        texts_origin = [word_tokenize(text_origin.lower().strip()) for text_origin in texts_origin]
        text_transfered = word_tokenize(text_transfered.lower().strip())
        return sentence_bleu(texts_origin, text_transfered) * 100

    def self_bleu_b(self, texts_origin, texts_transfered):
        assert len(texts_origin) == len(texts_transfered), 'Size of inputs does not match!'
        sum = 0
        n = len(texts_origin)
        for x, y in zip(texts_origin, texts_transfered):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_0(self, texts_neg2pos):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[0], texts_neg2pos):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu_1(self, texts_pos2neg):
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 500
        for x, y in zip(self.yelp_ref[1], texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    def yelp_ref_bleu(self, texts_neg2pos, texts_pos2neg):
        assert len(texts_neg2pos) == 500, 'Size of input differs from human reference file(500)!'
        assert len(texts_pos2neg) == 500, 'Size of input differs from human reference file(500)!'
        sum = 0
        n = 1000
        for x, y in zip(self.yelp_ref[0] + self.yelp_ref[1], texts_neg2pos + texts_pos2neg):
            sum += self.nltk_bleu([x], y)
        return sum / n

    
    def yelp_ppl(self, texts_transfered):
        texts_transfered = [' '.join(word_tokenize(itm.lower().strip())) for itm in texts_transfered]
        sum = 0
        words = []
        length = 0
        for i, line in enumerate(texts_transfered):
            words += [word for word in line.split()]
            length += len(line.split())
            score = self.yelp_ppl_model.score(line)
            sum += score
        return math.pow(10, -sum / length)

    def content_sim(self, texts_origin, texts_transferred):
        num_examples = len(texts_origin)
        total_sim = 0
        for i in range(num_examples):
            total_sim += self.find_similarity(texts_origin[0], texts_transferred[0])
        return total_sim / num_examples

    def make_example(self, sentence):
        sentence = sentence.lower()
        sentence = " ".join(self.tok.tokenize(sentence))
        sentence = self.sp.EncodeAsPieces(sentence)
        wp1 = Example(" ".join(sentence))
        wp1.populate_embeddings(self.sim.vocab)
        return wp1

    def find_similarity(self, s1, s2):
        with torch.no_grad():
            s1 = [self.make_example(x, self.sim) for x in s1]
            s2 = [self.make_example(x, self.sim) for x in s2]
            wx1, wl1, wm1 = self.sim.torchify_batch(s1)
            wx2, wl2, wm2 = self.sim.torchify_batch(s2)
            scores = self.sim.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
            return [x.item() for x in scores]

    
