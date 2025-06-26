library(sna)
library(numDeriv)

# read data
w1 <- as.matrix(read.csv(file="./Statistics/lnam/data/small/w1.csv"))
dimnames(w1) <- list(NULL,NULL)
w2 <- as.matrix(read.csv(file="./Statistics/lnam/data/small/w2.csv"))
dimnames(w2) <- list(NULL,NULL)
x <- as.matrix(read.csv(file="./Statistics/lnam/data/small/x.csv"))
dimnames(x) <- list(NULL,NULL)

beta <- read.csv(file="./Statistics/lnam/data/small/beta.csv")$x
nu <- read.csv(file="./Statistics/lnam/data/small/nu.csv")$x
e <- read.csv(file="./Statistics/lnam/data/small/e.csv")$x
y <- read.csv(file="./Statistics/lnam/data/small/y.csv")$V1

lnamfit <- lnam(y, x, w1, w2)
summary(lnamfit)
plot(lnamfit)
