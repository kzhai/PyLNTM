#!/usr/bin/env python

import cPickle, getopt, sys, time, re
import datetime, os;

import scipy
import scipy.io;
import nltk;
import numpy;
import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        word_correlation_file=None,
                        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,
                        number_of_topics=-1,

                        # parameter set 3
                        mu_alpha=0.,
                        sigma_alpha=0,
                        # alpha_beta=-1,
                        mu_beta=0.,
                        sigma_beta=0,
                        
                        # parameter set 4
                        optimization_method=None,
                        number_of_processes=1,
                        diagonal_covariance_matrix=False,
                        # disable_alpha_theta_update=False,
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    # parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      # help="the corpus name [None]")
    parser.add_option("--word_correlation_file", type="string", dest="word_correlation_file",
                      help="the word correlation file [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics", type="int", dest="number_of_topics",
                      help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");
                      
    # parameter set 3
    parser.add_option("--mu_alpha", type="float", dest="mu_alpha",
                      help="hyper-parameter for logistic normal distribution of topic [0.0]")
    parser.add_option("--sigma_alpha", type="float", dest="sigma_alpha",
                      help="hyper-parameter for logistic normal distribution of topic [1.0/number_of_topics]")
    parser.add_option("--mu_beta", type="float", dest="mu_beta",
                      help="hyper-parameter for logistic normal distribution of vocabulary [0.0]")
    parser.add_option("--sigma_beta", type="float", dest="sigma_beta",
                      help="hyper-parameter for logistic normal distribution of vocabulary [1.0/number_of_types]")
    
    # parameter set 4
    parser.add_option("--optimization_method", type="string", dest="optimization_method",
                      help="optimization method for logistic normal distribution");
    parser.add_option("--number_of_processes", type="int", dest="number_of_processes",
                      help="number of processes [1]");                      
    # parser.add_option("--diagonal_covariance_matrix", action="store_true", dest="diagonal_covariance_matrix",
                      # help="diagonal covariance matrix");
    # parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
                      # help="disable alpha (hyper-parameter for Dirichlet distribution of topics) update");
    
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    # parameter set 2
    assert(options.number_of_topics > 0);
    number_of_topics = options.number_of_topics;
    assert(options.training_iterations > 0);
    training_iterations = options.training_iterations;
    assert(options.snapshot_interval > 0);
    if options.snapshot_interval > 0:
        snapshot_interval = options.snapshot_interval;
    
    # parameter set 4
    optimization_method = options.optimization_method;
    if optimization_method == None:
        optimization_method = "L-BFGS-B";
    number_of_processes = options.number_of_processes;
    if number_of_processes <= 0:
        sys.stderr.write("invalid setting for number_of_processes, adjust to 1...\n");
        number_of_processes = 1;
    # diagonal_covariance_matrix = options.diagonal_covariance_matrix;
    # disable_alpha_theta_update = options.disable_alpha_theta_update;
    
    # parameter set 1
    # assert(options.corpus_name!=None);
    assert(options.input_directory != None);
    assert(options.output_directory != None);
    
    input_directory = options.input_directory;
    input_directory = input_directory.rstrip("/");
    corpus_name = os.path.basename(input_directory);
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    # Document
    train_docs_path = os.path.join(input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path, 'r');
    train_docs = [];
    for line in input_doc_stream:
        train_docs.append(line.strip().lower());
    print "successfully load all training docs from %s..." % (os.path.abspath(train_docs_path));
    
    # Vocabulary
    vocabulary_path = os.path.join(input_directory, 'voc.dat');
    input_voc_stream = open(vocabulary_path, 'r');
    vocab = [];
    for line in input_voc_stream:
        vocab.append(line.strip().lower().split()[0]);
    vocab = list(set(vocab));
    print "successfully load all the words from %s..." % (os.path.abspath(vocabulary_path));
    
    # parameter set 3
    alpha_mu = options.mu_alpha;
    alpha_sigma = options.sigma_alpha;
    if alpha_sigma <= 0:
        # alpha_sigma = 1.0/number_of_topics;
        alpha_sigma = 1.0
    assert(alpha_sigma > 0);
    beta_mu = options.mu_beta;
    beta_sigma = options.sigma_beta;
    if beta_sigma <= 0:
        # beta_sigma = 1.0/len(vocab);
        beta_sigma = 1.0
    assert(beta_sigma > 0);
    
    word_correlation_file = options.word_correlation_file;
    if word_correlation_file != None:
        if not os.path.exists(word_correlation_file):
            sys.stderr.write("word correlation file not found %s..." % word_correlation_file);
            sys.exit()
    
    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("lntm");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-K%d" % (number_of_topics);
    suffix += "-am%g" % (alpha_mu);
    suffix += "-as%g" % (alpha_sigma);
    suffix += "-bm%g" % (beta_mu);
    suffix += "-bs%g" % (beta_sigma);
    if optimization_method != None:
        suffix += "-%s" % (optimization_method.replace("-", "_"));
    if word_correlation_file != None:
        suffix += "-%s" % ("WordCorr");
    # suffix += "-HFO%s" % (optimization_method);
    # suffix += "-DCM%s" % (diagonal_covariance_matrix);
    # suffix += "-%s" % (resample_topics);
    # suffix += "-%s" % (hash_oov_words);
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));
        
    # store all the options to a file
    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("input_directory=" + input_directory + "\n");
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    if word_correlation_file != None:
        options_output_file.write("word_correlation=" + str(word_correlation_file) + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_mu=" + str(alpha_mu) + "\n");
    options_output_file.write("alpha_sigma=" + str(alpha_sigma) + "\n");
    options_output_file.write("beta_mu=" + str(beta_mu) + "\n");
    options_output_file.write("beta_sigma=" + str(beta_sigma) + "\n");
    # parameter set 4
    options_output_file.write("optimization_method=%s\n" % (optimization_method));
    # options_output_file.write("diagonal_covariance_matrix=%s\n" % (diagonal_covariance_matrix));
    options_output_file.write("number_of_processes=%d\n" % (number_of_processes));
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "input_directory=" + input_directory
    print "corpus_name=" + corpus_name
    if word_correlation_file != None:
        print "word_correlation_file=" + str(word_correlation_file)
    # parameter set 2
    print "training_iterations=%d" % (training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    print "number_of_topics=" + str(number_of_topics)
    # parameter set 3
    print "alpha_mu=" + str(alpha_mu)
    print "alpha_sigma=" + str(alpha_sigma)
    print "beta_mu=" + str(beta_mu)
    print "beta_sigma=" + str(beta_sigma)
    # parameter set 4
    print "optimization_method=%s" % (optimization_method)
    # print "diagonal_covariance_matrix=%s" % (diagonal_covariance_matrix)
    print "number_of_processes=%d" % (number_of_processes)
    print "========== ========== ========== ========== =========="
    
    import variational_bayes
    lntm_inferencer = variational_bayes.VariationalBayes(optimization_method);
    
    lntm_inferencer._initialize(train_docs, vocab, number_of_topics, alpha_mu, alpha_sigma, beta_mu, beta_sigma);
    
    if word_correlation_file != None:
        lntm_inferencer.import_word_correlation(word_correlation_file);
    
    # old_joint_log_likelihood = -sys.float_info.max;
    
    for iteration in xrange(training_iterations):
        new_joint_log_likelihood = lntm_inferencer.learning(number_of_processes);
     
        '''   
        if new_joint_log_likelihood<=old_joint_log_likelihood and numpy.abs((old_joint_log_likelihood-new_joint_log_likelihood)/old_joint_log_likelihood)>0.1:
            lntm_inferencer.export_beta(output_directory + 'exp_beta-' + str(lntm_inferencer._counter));
            model_snapshot_path = os.path.join(output_directory, 'model-' + str(lntm_inferencer._counter));
            cPickle.dump(lntm_inferencer, open(model_snapshot_path, 'wb'));
            break;
        old_joint_log_likelihood = new_joint_log_likelihood
        '''
                
        if (lntm_inferencer._counter % snapshot_interval == 0):
            lntm_inferencer.export_beta(os.path.join(output_directory, 'exp_beta-' + str(lntm_inferencer._counter)));

            model_snapshot_path = os.path.join(output_directory, 'model-' + str(lntm_inferencer._counter));
            cPickle.dump(lntm_inferencer, open(model_snapshot_path, 'wb'));

    model_snapshot_path = os.path.join(output_directory, 'model-' + str(lntm_inferencer._counter));
    cPickle.dump(lntm_inferencer, open(model_snapshot_path, 'wb'));
    
if __name__ == '__main__':
    main()
