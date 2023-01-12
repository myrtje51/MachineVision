#  /$$       /$$      /$$ /$$$$$$$$ /$$   /$$             /$$     /$$        /$$$$$$  /$$      /$$  /$$$$$$   /$$$$$$ 
# | $$      | $$$    /$$$| $$_____/| $$  | $$            | $$    | $$       /$$__  $$| $$  /$ | $$ /$$__  $$ /$$__  $$
# | $$      | $$$$  /$$$$| $$      | $$  | $$  /$$$$$$  /$$$$$$  | $$      | $$  \__/| $$ /$$$| $$| $$  \ $$| $$  \__/
# | $$      | $$ $$/$$ $$| $$$$$   | $$$$$$$$ /$$__  $$|_  $$_/  | $$      | $$ /$$$$| $$/$$ $$ $$| $$$$$$$$|  $$$$$$ 
# | $$      | $$  $$$| $$| $$__/   |_____  $$| $$  \ $$  | $$    | $$      | $$|_  $$| $$$$_  $$$$| $$__  $$ \____  $$
# | $$      | $$\  $ | $$| $$            | $$| $$  | $$  | $$ /$$| $$      | $$  \ $$| $$$/ \  $$$| $$  | $$ /$$  \ $$
# | $$$$$$$$| $$ \/  | $$| $$$$$$$$      | $$|  $$$$$$$  |  $$$$/| $$      |  $$$$$$/| $$/   \  $$| $$  | $$|  $$$$$$/
# |________/|__/     |__/|________/      |__/ \____  $$   \___/  |__/       \______/ |__/     \__/|__/  |__/ \______/ 
#                                                  | $$                                                               
#                                                  | $$                                                               
#                                                  |__/     

#####################Dependencies####################################
## load needed packages
library(Matrix)
library(MASS)
library(ggplot2)
library(openxlsx)
# for Step 1: linear mixed model (no SNPs)
library(lme4qtl) ##ATTENTION: You will need R 4.1.X to run this script without errors
# for Step 2: association tests
library(matlm) 
library(wlm) 
options(scipen = 999)
#####################Functions#######################################

##Inputs
#genotypes: Matrix with rows being lines and columns being the SNPs, the values are the alleles (1,2,3)
#trait: A vector with observances, binary or quantitative, with the same colnames and order as genotypes
#accession.type: A vector with respective Group of lines (e.g. virosa,saligna), same order as genotypes and trait cols
#phenotype.name: String, can be number or name
#snp.info: snp info in from of 3 columns (CHR,POS,QUAL)
#kinship: Kinship matrix (cov-var matrix of snp object)
#out.dir: Directory in which to store the results

##Outputs
#List with GWAS results: gassoc_gls, chr, pos


GWAS <- function(genotypes, trait, phenotype.name,snp.info, kinship, out.dir) {
  phenotype <- toString(phenotype.name)
  letkin <- kinship
  usemat <- genotypes
  selc <- !is.na(trait) #Selects the lines with an observation (removes lines that have an NA)
  trait.names <- trait[selc]
  use.trait <- trait[selc]
  print("Traits are selected.")
  print(paste("The phenotype ID is ", phenotype,".", sep=""))

  ## Filter in usemat and kinship object for mapping
  usemat <- usemat[selc,]
  letkin <- letkin[selc,selc]
  
  ## Filter again MAF 5%
  threshold <- round(nrow(usemat)*0.05, digits=0)
  selc2 <- apply(usemat==1,2,sum) >= threshold & apply(usemat==3,2,sum) >= threshold #1 is wild type allele, 3 is alternative allele
  print("SNPs falling within MAF >= 5%")
  print(table(selc2))
  usemat <- usemat[,selc2]
  #Save SNP positions from final SNP set
  
  chr <- snp.info[selc2,1]  ##Here we record the snp chromosome and chromosomal position 
  pos <- snp.info[selc2,2]
  
  
  
  print("Genotype matrix filtered and transformed.")
  
  phe.snp.cor <- cor(use.trait,usemat,use = "pairwise")  ##Here we correlate phenotypes with genotypes
  phe.snp.cor[is.na(phe.snp.cor)] <- 0  ###For NAs


### Make kinship
  snp.selc <- abs(phe.snp.cor)>0.3 & !is.na(phe.snp.cor) ##Here we select snps that have a correlation lower than 0.3
  usemat.pruned <- usemat #usemat[,snp.selc]
  ### start mapping by making decomposition
  ID <- rownames(letkin) ; length(ID)
  cbind(ID,use.trait)
  mod <- lme4qtl::relmatLmer(use.trait ~ (1|ID), relmat = list(ID = letkin))
  ##Calculate heritability
  herit.mod <- lme4qtl::VarProp(mod) #We calculate narrow sense heritability according to the authors of the lme4qtl script
  V <- lme4qtl::varcov(mod, idvar = "ID")
  V_thr <- V
  V_thr[abs(V) < 1e-10] <- 0
  decomp <- wlm::decompose_varcov(V, method = "evd", output = "all")
  W <- decomp$transform
  print("Decomposition of covariance matrix was performed.")
 
  ## make data object for mapping without any extra factors
  nnn <- rep(1,ncol(letkin))

  ### GWAS with kinship
  gassoc_gls <- matlm::matlm(use.trait ~ nnn, nnn, pred =  usemat.pruned, ids = rownames(W), transform = W, batch_size = 2000, verbose = 2,cores = 1,stats_full = F)
  gassoc_gls$trait.name <- phenotype
  gassoc_gls$trait.noobs <- length(use.trait)
  #lod <- rep(0,length(chr)) ##This step is done to save results of all SNPs again, also the ones that weren't tested
  #lod[snp.selc] <- -log10(gassoc_gls$tab$pval)
  lod <- -log10(gassoc_gls$tab$pval)
  #zscore <- rep(0,length(chr))
  #zscore[snp.selc] <- gassoc_gls$tab$zscore
  zscore <- gassoc_gls$tab$zscore
  mrkno <- which.max(lod) ##We save the most significantly associated snp for later use
 
  ###Save as integers
  ##We save values as integers to reduce the amount of disk space used when saving
  gassoc_gls$pval <- as.data.frame(lod*100000)
  gassoc_gls$zscore <- zscore*100000
  save(gassoc_gls,file=paste(out.dir,"/GWAS_result_",phenotype,".out",sep=""))
  save(herit.mod,file=paste(out.dir,"/Heritability_estimate_",phenotype,".out",sep=""))
  save(mod,file=paste(out.dir,"/Fitted_model_",phenotype,".out",sep=""))
  
  save(chr,file=paste(out.dir,"/GWAS_chromosome.out",sep=""))
  save(pos,file=paste(out.dir,"/GWAS_position.out",sep=""))
  cofac <- usemat[,mrkno]
  save(cofac,file=paste(out.dir,"/GWAS_cofac.out",sep="")) ##This is later use
  print("Results saved.")

}

########################START SCRIPT MYRTHE#########################
#setwd("/home/sarah/sarah2/LettuceKnow/BGI_Metabo_GWAS/")
ext.dir <- "/Users/myrthedehaan/Documents/UU internship Traitseeker/GWAS_data/output" ##The project directory where you save the results
#dir.create(ext.dir)
main.dir <- paste(ext.dir,"/GWAS_Results_test/",sep="") ###Output dir
dir.create(main.dir)


###INPUT


#### Load phenotype data Sarah
pheno <- y ##Y is created in Trying to make a phenotype data.R
phenot <- t(y)
colnames(phenot)<-unlist(phenot["bgi.id",])
phenot <-phenot[!row.names(phenot)=='bgi.id',]
phenot<-data.frame(t(unlist(phenot)))
pheno <- eval(parse(text=phenot))
base.dir <- paste(main.dir, "BGI_",sep="") ##Indicate which SNPset were used (BGI is a set of SNPs)

#Load genotype object for GWAS mapping
#Uncomment for serriola
geno <- usematF
geno <- as.matrix(usematF)
usemat <- eval(parse(text=usemat))
snp.info <- sat.snps[,c(1,2,3)] ##This is the positional info
usemat <- usemat[,c(-1,-2,-3)] ##The genotype matrix
usemat[usemat == 9] <- 2 ##We save unknown SNPs as Heterozygous allele because it is a minor amount
usemat <- as.matrix(t(usemat))

#Load kinship matrix
load("data/BGI_Sat_kinship.out") ##We generated the kinship matrix by taking the covariance of the SNPs
letkinF <- letkin[row.names(letkin)%in%eliminated$bgi.id,] ## Filter rows
letkinF2 <- letkinF[, (colnames(letkinF) %in% eliminated$bgi.id)] ## Filter columns
###Input phenotype
i <- commandArgs(trailingOnly = T) ##This is so that I can run GWAS in a loop from a bash parrallel script
i <- as.numeric(2)
trait <- rownames(phenot)[i] 
new.dir <- paste(base.dir,trait,sep="")
dir.create(new.dir)
print(phenot[i,])
log.file.w <- file(paste(new.dir,"/","BGI_",trait,"_warning.log",sep=""),open="wt") #I log error messages to see what went wrong
sink(file=log.file.w,type="message")

letkin_in <- letkin[names(pheno[i,]),names(pheno[i,])] #In case we do not have information for all lines with this phenotype
usemat_in <- usemat[names(pheno[i,]),] #In case we do not have information for all lines with this phenotype
 
#GWAS(genotypes = usematF, trait = as.vector(phenot[i,]), phenotype.name = trait, kinship=letkinF2, snp.info = snp.info, out.dir=new.dir)
GWAS(genotypes = usematF, trait = as.vector(as.numeric(phenot)), phenotype.name = "ARI2", kinship=letkinF2, snp.info = snp.info, out.dir="output")

sink(type="message")

close(log.file.w)

print(paste("GWAS finished. Phenotype is ",trait,sep=""))




