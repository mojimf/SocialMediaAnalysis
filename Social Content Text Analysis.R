#Installing packages----

##Sentiment analysis----
library(tidyverse)
library(tidytext)
library(devtools)
library(sentiment)
library(glue)
library(stringr)

##topic modeling----
library(tm)
library(tokenizers)
library(textstem)
library(SnowballC)
library(jsonlite)
library(dplyr)
library(stringr)
library(RColorBrewer)
library(wordcloud)
library(topicmodels)
library(ggplot2)
library(LDAvis)
library(servr)
library(textmineR)
library(syuzhet)

#data loading ----

setwd("C:/Users/m7918124/OneDrive - Virgin Media/VMO2 - Mojtaba/R studio/Social new")
RawData <- read.csv("C:/Users/m7918124/OneDrive - Virgin Media/VMO2 - Mojtaba/R studio/Social new/Appended with new sentiments v1.1.csv", header = T)

## data preparation ----

nrow(RawData)
RawData[,18:164] <- replace(RawData[,18:164],is.na(RawData[,18:164]),0)
TestSet <- RawData

##Merging the title and content fields and prepare subsets----

TestBind <- as.matrix(ifelse(TestSet$Title == TestSet$Content, TestSet$Title,
                      ifelse(TestSet$Title != TestSet$Content, paste(TestSet$Title,TestSet$Content), 'None')))
nrow(TestBind)
Binded <- cbind(TestBind,TestSet)
Posit <- subset(Binded, Sentiment_new == "positive")
Negat <- subset(Binded, Sentiment_new == "negative")
Nuet <- subset(Binded, Sentiment_new == "neutral")
nrow(Posit)+nrow(Negat)+nrow(Nuet)

#{You can define other data frames to be able to run the models for an specific tag an example could be: o2 <- subset(BInded, Tag.O2 == 1|...)}#

##amending the format and preparing the corpus----

reviewspo <- stringr::str_conv(Posit$TestBind, "UTF-8")
reviewsne <- stringr::str_conv(Negat$TestBind, "UTF-8")
reviewsnut <- stringr::str_conv(Nuet$TestBind, "UTF-8")
corppo <- Corpus(VectorSource(reviewspo))
corpne <- Corpus(VectorSource(reviewsne))
corpnut <- Corpus(VectorSource(reviewsnut))
write.csv(TestBind, "binded and ready.csv")
print(corppo[[1000]]$content)

## preparing the corpus of comments----

corpuspo <- tm_map(corppo, tolower)
corpuspo <- tm_map(corpuspo, removePunctuation)
corpusne <- tm_map(corpne, tolower)
corpusne <- tm_map(corpusne, removePunctuation)
corpusnut <- tm_map(corpnut, tolower)
corpusnut <- tm_map(corpusnut, removePunctuation)

#{To keep the O2 records, we did not remove the numbers from corpus. If you want to do that use this piece of code: corpus <- tm_map(corpus, removeNumbers) }

## lemmatising the corpus----
lemmapo <- tm_map(corpuspo, lemmatize_strings)
lemmane <- tm_map(corpusne, lemmatize_strings)
lemmanut <- tm_map(corpusnut, lemmatize_strings)

#print(corppo[[1000]]$content)

#{ to remove spaces, URL links and punctuation and remove specific words you can use these piece of codes:cleanset <- tm_map(lemmapo, removeWords, c('aapl', 'apple'))
#removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
#cleanset <- tm_map(cleanset, content_transformer(removeURL))
#removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x) 
#cleanset <- tm_map(cleanset, content_transformer(removeNumPunct))
#cleanset <- tm_map(cleanset, removeWords, stopwords()) 
#cleanset <- tm_map(cleanset, stripWhitespace)
#cleanset <- tm_map(cleanset, gsub, pattern = 'stocks', replacement = 'stock')
#cleanset <- tm_map(cleanset,toSpace,"[^[:graph:]]")
#cleanset <- tm_map(cleanset, stemDocument)
#cleanset <- tm_map(cleanset, stripWhitespace)}

## Tokenising the words ----
WordsTokenpo <- tokenize_words(lemmapo$content)
WordsTokenne <- tokenize_words(lemmane$content)
WordsTokennut <- tokenize_words(lemmanut$content)
print(WordsTokenpo[1000])
tokenspo <- as.list(WordsTokenpo)
tokensne <- as.list(WordsTokenne)
tokensnut <- as.list(WordsTokennut)

# Find associations of words----

#{Association extract the words that are related to a keyword with a defined corolation triger. For example if you wnat to see which words have correlation more than 0.1 with word "network", check the code below}
findAssocs(dtmne, terms = c("network"), corlimit = 0.1)

# Sentiment analysis---- 

#{ We used the raw text for sentiment analysis to avoid any misleading result. Sometimes lemmatisation mislead the model}

Sentiment_new <- sentiment(Binded$TestBind[1:nrow(Binded)])
#sentiments1 <- sentiment(Binded$TestBind[20590:20599])


for(i in seq(from=1, to=nrow(Binded), by=100)){
  sentiments <- rbind(Sentiment_new,sentiment(Binded$TestBind[i:min((i+99),nrow(Binded))]))
  print(i)
}


#checks for the results: if you decided to run the model for chunks of data, you can check the results here.
#nrow(sentiments1)+nrow(sentiments)+nrow(sentiments2)+nrow(sentiments3)+nrow(sentiments4)+nrow(sentiments5)+nrow(sentiments6)
#table(sentiments$polarity[1:100])
#sentiments[11:15,]

#appending the tables: if you decided to run the model for chunks of data, you can bind the results here.

sentfin <- do.call("rbind", list(sentiments,sentiments1,sentiments2,sentiments3,sentiments4,sentiments5,sentiments6))
table(sentfin$polarity)
write.csv(sentfin, "sentiments.csv")
a <- Binded
names(a)[names(a) == "TestBind"] <- "text"
merged <- merge(a,sentfin, by = "text")
columns(merged)


#### term matrix preparation, frequency and removing spares----

##PO
dtmpo <- DocumentTermMatrix(WordsTokenpo, control = list(lemma = T, removePunctuation =T, removeNumbers=F,stopwords=T,tolower=T,wordLengths=c(1,Inf)))
findFreqTerms(dtmpo,lowfreq = 10000)
dtmspo <- removeSparseTerms(dtmpo, 0.99)
RawSumspo <- apply(dtmspo,1,FUN=sum)
dtmspo <- dtmspo[RawSumspo!=0,]
outputpo <- as.matrix(dtmspo)
nrow(outputpo)
ncol(outputpo)
#Ne
dtmne <- DocumentTermMatrix(WordsTokenne, control = list(lemma = T, removePunctuation =T, removeNumbers=F,stopwords=T,tolower=T,wordLengths=c(1,Inf)))
findFreqTerms(dtmne,lowfreq = 10000)
dtmsne <- removeSparseTerms(dtmne, 0.99)
RawSumsne <- apply(dtmsne,1,FUN=sum)
dtmsne <- dtmsne[RawSumsne!=0,]
outputne <- as.matrix(dtmsne)
nrow(outputne)
ncol(outputne)
#Nut
dtmnut <- DocumentTermMatrix(WordsTokennut, control = list(lemma = T, removePunctuation =T, removeNumbers=F,stopwords=T,tolower=T,wordLengths=c(1,Inf)))
findFreqTerms(dtmnut,lowfreq = 10000)
dtmsnut <- removeSparseTerms(dtmnut, 0.99)
RawSumsnut <- apply(dtmsnut,1,FUN=sum)
dtmsnut <- dtmspo[RawSumsnut!=0,]
outputnut <- as.matrix(dtmsnut)
nrow(outputnut)
ncol(outputnut)

#### frequency table----
#po
dtmsNewpo <- as.matrix(dtmspo)
frequencypo <- colSums(dtmsNewpo)
frequencypo <- sort(frequencypo, decreasing = T)
DocLengthpo <- rowSums(dtmsNewpo)
dtm_frequencypo <- data.frame(word = names(frequencypo),freq=frequencypo)
words <- names(frequencypo)
wordcloud(words[1:100], frequencypo[1:100], rot.per=0.15, random.order =F, scale = c(5,0.5), random.color = F, colors=brewer.pal(8,"Dark2"))
write.csv(dtm_frequencypo, "frequencypo.csv")
#ne
dtmsNewne <- as.matrix(dtmsne)
frequencyne <- colSums(dtmsNewne)
frequencyne <- sort(frequencyne, decreasing = T)
DocLengthne <- rowSums(dtmsNewne)
dtm_frequencyne <- data.frame(word = names(frequencyne),freq=frequencyne)
words <- names(frequencyne)
wordcloud(words[1:100], frequencyne[1:100], rot.per=0.15, random.order =F, scale = c(5,0.5), random.color = F, colors=brewer.pal(8,"Dark2"))
write.csv(dtm_frequencyne, "frequencyne.csv")
#nut
dtmsNewnut <- as.matrix(dtmsnut)
frequencynut <- colSums(dtmsNewnut)
frequencynut <- sort(frequencynut, decreasing = T)
DocLengthnut <- rowSums(dtmsNewnut)
dtm_frequencynut <- data.frame(word = names(frequencynut),freq=frequencynut)
words <- names(frequencynut)
wordcloud(words[1:100], frequencynut[1:100], rot.per=0.15, random.order =F, scale = c(5,0.5), random.color = F, colors=brewer.pal(8,"Dark2"))
write.csv(dtm_frequencynut, "frequencynut.csv")


#### emotion analysis

#### emotions ----
# use the cleaned text for emotion analysis 

text_cleanpo <- data.frame(text = sapply(corppo, as.character), stringsAsFactors = FALSE)
text_cleanne <- data.frame(text = sapply(corpne, as.character), stringsAsFactors = FALSE)
text_cleannut <- data.frame(text = sapply(corpnut, as.character), stringsAsFactors = FALSE)


# each row: a document (a tweet); each column: an emotion
#po
emotion_matrixpo <- get_nrc_sentiment(text_cleanpo$text)
text_cleanpo$text[10]
emotion_matrixpo[10,]
nrow(emotion_matrixpo)
#ne
emotion_matrixne <- get_nrc_sentiment(text_cleanne$text)
text_cleanne$text[10]
emotion_matrixne[10,]
nrow(emotion_matrixne)
#nut
emotion_matrixnut <- get_nrc_sentiment(text_cleannut$text)
text_cleannut$text[10]
emotion_matrixnut[10,]
nrow(emotion_matrixnut)

# Matrix Transpose
# so each row will be an emotion and each column will be a tweet

tdpo <- data.frame(t(emotion_matrixpo)) 
tdne <- data.frame(t(emotion_matrixne))
tdnut <- data.frame(t(emotion_matrixnut))
#The function rowSums computes column sums across rows for each level of a grouping variable.
td_newpo <- data.frame(rowSums(tdpo))
td_newne <- data.frame(rowSums(tdne))
td_newnut <- data.frame(rowSums(tdnut))


#Transformation and cleaning
names(td_newpo)[1] <- "count"
td_newpo
td_newpo <- cbind("sentiment" = rownames(td_newpo), td_newpo)
td_newpo
rownames(td_newpo) <- NULL
nrow(td_newpo)

# emotion Visualisation
library("ggplot2")
qplot(sentiment, data=td_newpo, weight=count, geom="bar",fill=sentiment)+ggtitle("Text emotion and sentiment")

# extracting the emotions
po <- cbind(Posit,emotion_matrixpo)
ne <- cbind(Negat,emotion_matrixne)
nut <- cbind(Nuet, emotion_matrixnut)
nrow(po)+nrow(ne)+nrow(nut)
SeEm <- rbind(po,ne,nut)
write.csv(SeEm[order(SeEm$X), ],'sentiments and emotions.csv')

#### Topic modeling----

#po
dtmpo
topic_num = 10
term_num = 5
rowTotals <- apply(dtmpo , 1, sum) #Find the sum of words in each Document (tweet)
dtm  <- dtmpo[rowTotals> 0, ] #remove all docs without words
lda <- LDA(dtm, k = topic_num) # find k topics
term <- terms(lda, term_num) # first term_num terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))

### model interpretation

k=7
iter=100
ldapo <- LDA(dtmspo,k, method="Gibbs", control=list(iter=iter, seed=023))
ldaOuttermspo <- as.matrix(terms(ldapo, 20))
ldapoTerms <- as.matrix(terms(ldapo,20))
matpo <- as.matrix(ldaOuttermspo)
write.csv(matpo,file="topicspo.csv")

k=7
iter=100
ldane <- LDA(dtmsne,k, method="Gibbs", control=list(iter=iter, seed=023))
ldaOuttermsne <- as.matrix(terms(ldane, 20))
ldaneTerms <- as.matrix(terms(ldane,20))
matne <- as.matrix(ldaOuttermsne)
write.csv(matne,file="topicsne.csv")

k=7
iter=100
ldanut <- LDA(dtmsnut,k, method="Gibbs", control=list(iter=iter, seed=023))
ldaOuttermsnut <- as.matrix(terms(ldanut, 20))
ldanutTerms <- as.matrix(terms(ldanut,20))
matnut <- as.matrix(ldaOuttermsnut)
write.csv(matnut,file="topicsnut.csv")

### documents vs topics map(keep it for future development)----

ldaOutTopicspo <- data.frame(topics(ldapo))
nrow(ldaOutTopicspo)
ldaOutTopicspo$index <- as.numeric(row.names(ldaOutTopicspo))
Posit$General_index <- row.names(Posit)
row.names(Posit) <- NULL
Posit$index <- as.numeric(row.names(Posit))
nrow(Posit)
datawithtopicpo <- merge(Posit, ldaOutTopicspo, by='index',all.x=TRUE)
#datawithtopicpo <- datawithtopicpo[order(datawithtopicpo$index), ]
write.csv(datawithtopicpo,file="topicspo_inaddition_documents.csv")

ldaOutTopicsne <- data.frame(topics(ldane))
nrow(ldaOutTopicsne)
ldaOutTopicsne$index <- as.numeric(row.names(ldaOutTopicsne))
Negat$General_index <- row.names(Negat)
row.names(Negat) <- NULL
Negat$index <- as.numeric(row.names(Negat))
nrow(Negat)
datawithtopicne <- merge(Negat, ldaOutTopicsne, by='index',all.x=TRUE)
write.csv(datawithtopicne,file="topicsne_inaddition_documents.csv")

ldaOutTopicsneu <- data.frame(topics(ldanu))
nrow(ldaOutTopicsneu)
ldaOutTopicsneu$index <- as.numeric(row.names(ldaOutTopicsneu))
Nuet$General_index <- row.names(Nuet)
row.names(Nuet) <- NULL
Nuet$index <- as.numeric(row.names(Nuet))
nrow(Nuet)
datawithtopicneu <- merge(Nuet, ldaOutTopicsneu, by='index',all.x=TRUE)
write.csv(datawithtopicneu,file="topicsneu_inaddition_documents.csv")

### documents in each topic

#topicProbabilitiespo <- as.data.frame(ldapo@gamma)

