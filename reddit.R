# Data Mining - Project Code
# Donald Chesworth, Yuning Ling, Zhuoyang Zhou, Sharath Pingula
#--------- Recommendation System using Topic Modeling ------
library(ggplot2)
library(dplyr)
library(reshape2)
library(tm)
library(topicmodels)
library(slam)
library(lsa)
library(qdapRegex)
setwd('/Users/homw/Documents/MSDS16/DataMining/reddit/')

#------- Functions ------------------------------
#creates DTM using tm package functions
get.dtm <- function(author.doc){
  corpus <- author.doc[,"s"]
  
  #Transform all reddits into a corpus
  corpus <- as.data.frame(corpus)
  corpus <- VCorpus(DataframeSource(corpus))
  # clean and compute tf
  
  corpus.clean = tm_map(corpus, stripWhitespace)    # remove extra whitespace
  corpus.clean = tm_map(corpus.clean, removeNumbers)  # remove numbers
  corpus.clean = tm_map(corpus.clean, removePunctuation)    # remove punctuation
  corpus.clean = tm_map(corpus.clean, content_transformer(tolower))   # ignore case
  corpus.clean = tm_map(corpus.clean, removeWords, stopwords("english"))# remove stop words
  
  corpus.clean = tm_map(corpus.clean, stemDocument) # stem all words
  #remove high freq words - created specifically for reddit
  corpus.clean <- tm_map(corpus.clean, removeWords, as.vector(highfreq)) 
  
  #inspect(corpus.clean[10:12])
  corpus.tf = DocumentTermMatrix(corpus.clean, control = list(weighting = weightTf))
  
  #inspect(corpus.clean[1:2])
  #Remove empty documents
  #corpus.tf <- corpus.tf[row_sums(corpus.tf, na.rm=T)>0,]
  return(corpus.tf)
}

#----------------- Main code ----------
red <- read.csv('redfinal.csv', header = T)
highfreq <- as.vector(read.csv("highfreq.csv")[,2]) #Created for Reddit!
length(unique(red$author))

#Subset only required fields
red1 <- red[,c("name","subreddit","author","body", "distinguished", "parent_id")]
rm(red) #remove raw data, taking up a lot of space

#Select random 100,000 posts. We have tried with 1million and 0.5 million posts. But it has taken 2 days and could not complete one topic modeling task out of two. We limited ourselves to 100,000 posts finally
red5k <- red1[sample(nrow(red1),100000),]

t <- table(red5k$subreddit) # 7373 subreddits
t[order(t, decreasing = T)][1:10] 
length(t[t>200]) # 84 subreddits has morethan 200 posts!
sr.names <- names(t[t>100]) # All the SRs with morethan 100 posts
red5k1 <- red5k
red5k <- red5k[red5k$subreddit %in% sr.names,] # Only SRs with more than 100 posts

c <- red5k %>% group_by(author) %>% summarize(total.count=n()) #Count of posts from each user
sc <- red5k %>% group_by(subreddit) %>% summarise(total.count = n()) #Count of posts in SRs

# Building topics for Authors and Subreddits separately. Match those topics to find most matching subreddits of a given author based on the topics his posts contained.

# Step1 # ----- DTM on Authors
#Prepare author documents - all posts of an author are aggregated and called as a document
author.doc <- red5k %>% group_by(author) %>% summarize(s=paste(body, collapse=" "))
author.doc <- as.data.frame(author.doc)
author.doc$s <- sapply(author.doc$s,function(row) iconv(row, "latin1", "ASCII", sub="")) 
author.doc$s <- rm_url(author.doc$s) #Removes url addresses
author.doc$s <- rm_emoticon(author.doc$s) #Removes url addresses
# Keep the workspace clean for faster exec
#rm(c,sc)
#Get the doc term matrix - Tf scoring
tm <- proc.time()
author.dtm <- get.dtm(author.doc)
author.dtm #Check the num of terms
proc.time() - tm

zero.rows <- which(row_sums(author.dtm)<=0)
#remove zeros row authors
author.doc <- author.doc[-zero.rows,]

#Step2 # ------ DTM on Subreddits
# Composition of subreddits (sr) in the data
#Select all the SRs that have more than n posts in them
#sr1 <- red5k[red5k$subreddit %in% sr.names,]

# Aggregate the posts by subreddit. All the posts belong to SR should be grouped
tm <- proc.time()
sr.doc <- red5k %>% group_by(subreddit) %>% summarize(s=paste(body, collapse=" "))
sr.doc <- as.data.frame(sr.doc)
sr.doc$s <- sapply(sr.doc$s,function(row) iconv(row, "latin1", "ASCII", sub="")) 
sr.doc$s <- rm_url(sr.doc$s) #Removes url addresses
sr.doc$s <- rm_emoticon(sr.doc$s) #Removes url addresses
#Create DTM
sr.dtm <- get.dtm(sr.doc)
sr.dtm #Check the num of terms
proc.time() - tm

zero.rows <- which(row_sums(sr.dtm)<=0)
#remove zeros row authors
sr.doc <- sr.doc[-zero.rows,]

# Step3 # Ensure same terms in both the matrices. It is mandatory for cosine similarity

#Avoid this step if author.dtm and sr.dtm has same number of terms. It takes a lot of space.
# author.dtm <- as.matrix(author.dtm)
# sr.dtm <- as.matrix(sr.dtm)
sr.dtm <- sr.dtm[,colnames(sr.dtm) %in% colnames(author.dtm)]
author.dtm <- author.dtm[,colnames(author.dtm) %in% colnames(sr.dtm)]

# Step4 # Run topics on Authors 
#Remove empty documents
author.dtm <- author.dtm[row_sums(author.dtm, na.rm=T)>0,]
# Run topic model on authors using LDA
tm <- proc.time()
author.topics = LDA(author.dtm, 50)
proc.time() - tm
#Each author can now be represented as a vector of 10 topics
author.doc.prob = as.data.frame(author.topics@gamma) #cool.
terms(author.topics,10)
#topics(author.topics)

# Step5 # Run topic model on SRs using LDA
#Remove empty documents
sr.dtm <- sr.dtm[row_sums(sr.dtm, na.rm=T)>0,]
sr.topics = LDA(sr.dtm, 50)
#Each SR can now be represented as a vector of 15 topics
sr.doc.prob = as.data.frame(sr.topics@gamma) #cool.
terms(sr.topics,10)
#topics(sr.topics)

#Step6 #  Get the topic that is best representing a SR
t <- as.data.frame(topics(sr.topics)) #Top topic in each SR
sr.doc$subreddit <- as.character.factor(sr.doc$subreddit)
sr.top.topic <- as.data.frame(cbind(sr.doc[,'subreddit'],t[,1]))
names(sr.top.topic) <- c('subreddit','top_topic')

#Step7 # Find the best matching Author topics for each SR
# for every SR -> its top topic will be matched against all Author topics and the best matched one will be selected.
sr.topic.word <- as.data.frame(sr.topics@beta) #Topic word matrix of Subreddit
author.topic.word <- as.data.frame(author.topics@beta)

#Need for troubleshooting
# st1 <- sr.topic.word[1,]
# at1 <- author.topic.word[1,]
# length(st1[st1 >= -7])
# length(at1[at1 >= -7])

# Binarize word topic probabilities of Author topics
for (a in 1:author.topics@k){
   at1 <- author.topic.word[a,]
#   at1 <- (at1-min(at1))/1e5
  q9 <- quantile(at1,0.999)
  names(q9) <- "q"
  at1 <- ifelse(at1 > q9$q,1,0)
  author.topic.word[a,] <- at1
  cat(a)
}
#length(at1[at1>0])
# Binarize word topic probabilities of SR topics
for (a in 1:sr.topics@k){
   st1 <- sr.topic.word[a,]
#   st1 <- (st1-min(st1))/1e5
  q9 <- quantile(st1,0.999)
  names(q9) <- "q"
  st1 <- ifelse(st1 > q9$q,1,0)
  sr.topic.word[a,] <- st1
  cat(a)
}

# Initializing variables
max.match <- 1:author.topics@k
sr.auth.match1 <- 1:sr.topics@k
sr.auth.match2 <- 1:sr.topics@k

# Calculates cosine similarity between a and b
cos.sim=function(a, b){
  #return(sum(a*b)/sqrt(sum(a^2))*sqrt(sum(b^2)))
  return(sum(a*b))
}

# For SR get the top 2 topics from author topics using cosine similarity
tm <- proc.time()
for (n in 1:sr.topics@k){
#   st <- sr.top.topic[n,"top_topic"] #get the top topic number
#   st <- as.integer(as.character(st))
  st1 <- sr.topic.word[n,] #Get the words vector of sr topic
  for (a in 1:author.topics@k){
    # Find the similarity between SR topic and Author topic
    max.match[a] <- cos.sim(as.numeric(st1), as.numeric(author.topic.word[a,]))
  }
#   sr.auth.match1[n] <- sort(max.match, decreasing = TRUE)[1]
#   sr.auth.match2[n] <- sort(max.match, decreasing = TRUE)[2]
  
  # Get topic number of best and 2nd best matched topics for Every SR
  sr.auth.match1[n] <- which.max(max.match)
  sr.auth.match2[n] <- which.max(max.match[max.match!=max(max.match)])
  cat(n)
}
proc.time() - tm

#Add top 2 matching topics for Every Subreddit from Author topics
sr.auth <- as.data.frame(cbind(1:sr.topics@k,sr.auth.match1,sr.auth.match2))
names(sr.auth) <- c('top_topic','match1','match2')
sr.auth$top_topic <- as.factor(sr.auth$top_topic)
sr.top.topic <- inner_join(sr.top.topic, sr.auth)

table(sr.auth$match1)
table(sr.top.topic$match1)
table(sr.top.topic$match2)

#arrange sr.top.topic file as per the popular subreddits
s <- inner_join(sr.top.topic, sc)
s <- s[order(s$total.count, decreasing = T),]
sr.top.topic.sorted <- s
table(sr.top.topic.sorted$match1)
#Now we have Author topic matrix and a linked list of subreddits to author topics

# -------- Part 1 Completed -----------------



# --------- Part 2 - Recommendation System ----------
#Create top topics of authors
author.top <- as.data.frame(cbind(1:nrow(author.doc.prob),topics(author.topics)))
names(author.top) <- c('author','top_topic')

#Recommendation System idea 1
#Every user is vector of topics. Get the best (k=2) topics (max probability) for a target author. 
#Get the subreddits that match with best topics of the author
#Filter out existing, recommend the balance
#---The gist of Idea1 ---
#--- We are suggesting an author the best matched subreddits based on type of topics his posts contained 

# Recommedations using target author's topics alone
samp <- sample(nrow(author.doc),100)
target_auth <- author.doc.prob[samp,]

check.rec <- as.data.frame(matrix(NA,length(samp),3))
names(check.rec) <- c('author','actual','recommended')
check.rec$author <- author.doc[samp,'author']
for(i in 1:length(samp)){
  check.rec[i,"actual"] <- paste0(unique(red5k[red5k$author == author.doc[samp[i],"author"],'subreddit']),sep = ' | ', collapse='')
  
  # check.rec[i,'recommended'] <-  paste0(sr.top.topic.sorted[sr.top.topic.sorted$match1==author.top[author.top$author==samp[i],'top_topic'],'subreddit'],sep = ' | ', collapse = '')
  top1 <- which.max(author.doc.prob[samp[i],])
  top2 <- which.max(author.doc.prob[samp[i],-top1])
  
  check.rec[i,'recommended'] <-  paste0(sr.top.topic.sorted[sr.top.topic.sorted$match1 %in% c(top1,top2),'subreddit'],sep = ' | ', collapse = '')
}
write.csv(check.rec, 'recommends1.csv')

#Recommendation System idea 2
# For every target user, get the neighbors from Author topic matrix
# Get the topics of those neighbors 
# Get the subreddits closest to those topics from the linked list of SR - Auth.topics
# Filter out the topics target user has registered with
# Recommend the balance topics

#---The gist of Idea2 ---
#--- We are suggesting an author the best matched subreddits based on type of topics his neighbors' posts contained 

# We use SVD - Singular Value Decomposition to get author neighborhood space for a target author

# Author - Topic Matrix
max(author.doc.prob[1,])
# a1 <- author.doc.prob
# a1[a1<0.1] <- 0
svd.auth <- lsa(author.doc.prob,10)

svd.auth$sk #Eigen vector
dim(svd.auth$dk) # Topic Space
dim(svd.auth$tk) # Author Space

#Create SVD Matrix
sk <- array(dim = c(10,10), 0)
diag(sk) <- svd.auth$sk
#sk

# convert the target authors into SVD author space
target_auth_svd <- as.matrix(target_auth) %*% svd.auth$dk %*% solve(sk)

#Get nearest neighbors for each target author
# Create a distance data frame with all the Authors
author = 0
similar = 0

#first remove the target authors from the authro space, since we have taken target authors from existing authors set.
# When a new author comes, this step would not necessary
author.space <- as.data.frame(svd.auth$tk)
author.space <- author.space[-samp,]

ta.neighbors <- matrix(NA,nrow = nrow(target_auth),ncol = 2)
for (ta in 1:nrow(target_auth)){
  for(a in 1:nrow(author.space)) {
    similar[a] = cosine(as.vector(target_auth_svd[ta,]), as.vector(as.matrix(author.space[a,])))
    author[a] = a
  }
  
  result <- data.frame(author, similar)
  #get top k closest neighbors of target author
  ta.neighbors[ta,1:2] <- result[order(similar, decreasing = TRUE),'author'][1:2]
  cat(ta)
}

#Suggest recommendations from (k=2) nearest neighbors
check.rec.svd <- as.data.frame(matrix(NA,length(samp),3))
names(check.rec.svd) <- c('author','actual','recommended')
check.rec.svd$author <- author.doc[samp,'author']

for(i in 1:nrow(target_auth)){
  check.rec.svd[i,"actual"] <- paste0(unique(red5k[red5k$author == author.doc[samp[i],"author"],'subreddit']),sep = ' | ', collapse='')
  rec <- author.top[author.top$author %in% ta.neighbors[i,], 'top_topic']
  
  check.rec.svd[i,"recommended"] <- paste0(sr.top.topic.sorted[sr.top.topic.sorted$match1 %in% unique(rec),'subreddit'], sep=' | ', collapse = '')
}

write.csv(check.rec.svd, 'recommends2_svd.csv')

# --------- Recommendation System code Completed -------------------


#_______ Extracting 1 million posts from a sql database of 55 million posts --

## Much of the code was found here: http://www.r-bloggers.com/r-and-sqlite-part-1/

##### LIBRARIES #####
library(RSQLite)
library(DBI)
library(sqldf)

##### FUNCTION #####
sqlite.subset.reditors = function(new.db.name, high.comments, low.comments) {
  
  # Create a dataframe of only authors and number of posts
  con <- dbConnect(RSQLite::SQLite(), dbname='database.sqlite')
  author.cmd <- 'SELECT author, count(author) FROM May2015 group by author'
  grouped.authors <- dbGetQuery(con, author.cmd)
  
  # Get a list of authors that have a certain number of comments
  subset.authors <- grouped.authors$author[grouped.authors$`count(author)` < high.comments & grouped.authors$`count(author)` > low.comments]
  subset.authors <- paste(shQuote(subset.authors), collapse=", ")
  
  # Create a subset table of the original table with the desired authors, then a database
  subset.cmd <- paste0("SELECT * FROM May2015 WHERE author in (", subset.authors, ")")
  subsetReditors <- dbGetQuery(con, subset.cmd)
  attach.cmd <- paste0('ATTACH ', "'", new.db.name, "'", ' AS new')
  sqldf(attach.cmd)
  sqldf('CREATE TABLE May2015 AS SELECT * FROM subsetReditors', dbname = new.db.name)
  
  print("file created")
  
  return(subsetReditors)
}


##### EXECUTION #####

# Create the index on the author table
con <- dbConnect(RSQLite::SQLite(), dbname='database.sqlite')
dbGetQuery(con, "CREATE INDEX index_auth ON May2015 (author)")

# Create the subset of authors database, and save the dataframe
subsetTable <- sqlite.subset.reditors('redditTop.sqlite', 575, 500)
#-------------
