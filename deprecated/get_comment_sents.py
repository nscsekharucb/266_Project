#
# Routine to extract comments from a given file
#
def get_comment_sents(filename):
    comment_blocks = comment_parser.extract_comments(filename)
    comment_sents = []
    
    #
    # Skip copyright section
    #
    for comment_block in comment_blocks[1:]:
        #
        # Remove any special characters
        #
        comment_text = comment_block._text
        comment_text = comment_text.replace('*', '')
        comment_text = comment_text.replace('\n', '')
        comment_text = comment_text.replace('\t', '')
        
        for sent in tokenize.sent_tokenize(comment_text):
            comment_sents.append(sent)

    return comment_sents