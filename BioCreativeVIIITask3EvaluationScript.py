import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import logging as log
import os.path
from pathlib import Path
import sys
import numpy as np

class Findings(object):
    def __init__(self, Truth: pd.DataFrame, Preds: pd.DataFrame):
        """
        Representation of the annotations manually labeled for the gold standard and for the automatically generated predictions
        :param: Truth: a dataframe validated by the evaluator, contains the manual annotations of the observations
        :param: Preds: a dataframe validated by the evaluator, contains the predicted annotations for the observations
        :return: compute the TPs, TNs, FPs, FNs (normalization only plus exact/overlapping extraction + normalization if attempted)
                 create the DataFrame with all annotations aligned for verification (in self.observations)
        """
        def getHPOID2TERM()->pd.DataFrame:
            try:
                PATH_HPOID2TERM = str(Path(os.path.abspath(os.curdir)).parent) + "/data/HP2Terms.tsv"
                if os.path.exists(PATH_HPOID2TERM):
                    hpo2term = pd.read_csv(PATH_HPOID2TERM, sep="\t")
                    hpo2term.set_index("HPO_ID", verify_integrity=True, inplace=True)
                    return hpo2term
                else:
                    return None
            except Exception as e:
                log.error(f"Impossible to read the HPO2Terms.tsv file, mapping unavailable: {e}")
                return None
        self.hpo2term = getHPOID2TERM()
        self.truth = Truth
        self.pred = Preds
        self.onlyNormalization = self.pred["Spans"].isnull().all()
        obsInTruth = self.truth["ObservationID"].unique()
        obsInTruth.sort()
        self.TNs, self.TPs, self.FNs, self.FPs = 0, 0, 0, 0  # basic observations for the normalization only
        if self.onlyNormalization:
            log.debug(f"Findings have NOT been extracted, just normalized.")
            cols = ["ObservationID", "Text", "Truth HPO ID", "Truth Spans", "Predicted HPO ID", "Predicted Spans", "Normalization"]
            if self.hpo2term is not None:
                cols.insert(3, "Truth HPO Term")
                cols.insert(6, "Predicted HPO Term")
            self.observations = pd.DataFrame(columns=cols)
        else:
            log.debug(f"Findings have been extracted, will also compute extraction performance.")
            self.eTNs, self.eTPs, self.eFNs, self.eFPs = 0, 0, 0, 0  # basic observations for the extact extraction and normalization
            self.oTNs, self.oTPs, self.oFNs, self.oFPs = 0, 0, 0, 0  ##basic observations for the overlapping extraction and normalization
            cols = ["ObservationID", "Text", "Truth HPO ID", "Truth Spans", "Truth Spans Text", "Predicted HPO ID", "Predicted Spans", "Predicted Spans Text", "Normalization", "Exact Extraction & Normalization", "Overlapping Extraction & Normalization"]
            if self.hpo2term is not None:
                cols.insert(3, "Truth HPO Term")
                cols.insert(7, "Predicted HPO Term")
            self.observations = pd.DataFrame(columns=cols)
        #I align the annotations observation per observation and add the alignment in self.observations
        for obsID in obsInTruth:
            #log.debug(f"Extracting annotations for Observation: [{obsID}]")
            self.__addObservation(self.truth[self.truth["ObservationID"] == obsID], self.pred[self.pred["ObservationID"] == obsID])

    def __getPreferredTerm(self, HPOID:str)->str:
        """
        Given an HPO ID returns the preferred term
        :param HPOID:
        :return: preferred term, "HPO_TERM_UNKNOWN" if the mapping is unavailable or the HPO ID is unknown
        """
        if self.hpo2term is not None and HPOID in self.hpo2term.index:
            return self.hpo2term.loc[HPOID]["Preferred_Term"]
        else:
            return "HPO_TERM_UNKNOWN"

    def __addObservation(self, truthAnnotations:pd.DataFrame, predAnnotations:pd.DataFrame):
        """
        Main function, aligns the annotations predicted and annotated for 1 observation
        :param: truthAnnotations, a DF that only contains the annotations for the observation to be added
        :param: predAnnotations, another DF that only contains the predictions for the observation to be added
        :return: add the alignments of the annotations/predictions to the main DF self.observations
        """
        isOnlyTruthNormalFinding, trueFindings = self.__formatTruthFindings(truthAnnotations)
        #uncomment if you need to debug a specific observation
        # if trueFindings.iloc[0]["ObservationID"] == "00db0b8c6892cbdb8fc3b02fed791b9d":
        #     log.debug("Stopped here.")
        isOnlyPredNormalFinding, predFindings = self.__formatPredictedFindings(predAnnotations)
        self.__addTNs(isOnlyTruthNormalFinding, trueFindings, isOnlyPredNormalFinding, predFindings)
        self.__addFPs(isOnlyTruthNormalFinding, trueFindings, isOnlyPredNormalFinding, predFindings)
        self.__addFNs(isOnlyTruthNormalFinding, trueFindings, isOnlyPredNormalFinding, predFindings)
        self.__addPossibleTPs(isOnlyTruthNormalFinding, trueFindings, isOnlyPredNormalFinding, predFindings)

    def __formatTruthFindings(self, truthAnnotations:pd.DataFrame) -> list:
        """
        Remove the negated normal findings from the list of findings annotated, leaving only key findings
        :param: truthAnnotations, a DF containing all manually labeled annotations for a specific observation
        :return: a bool at True if the observation contains only normal findings
        :return: the dataFrame with only key findings, if no key findings where mentioned in the observation,
                 just return one line with None in the fields
        """
        # some observation have no findings annotated (all findings are normal)
        keyFindings = truthAnnotations[truthAnnotations["HPO Term"].str.startswith("HP:") & truthAnnotations["Polarity"].isnull() & truthAnnotations["Spans"].notnull()]
        if len(keyFindings)==0:
            # if an observation has no key finding, we just create a normal findings (we ignore all negated HPO terms if any)
            normalFinding = pd.DataFrame([truthAnnotations.iloc[0].copy()])
            normalFinding["HPO Term"] = np.nan
            normalFinding["Polarity"] = np.nan
            normalFinding["Spans"] = np.nan
            return [True, normalFinding]
        else:
            # one or multiple HPO terms are found
            return [False, keyFindings.copy()]

    def __formatPredictedFindings(self, predictedAnnotations:pd.DataFrame)-> list:
        """
        RemoveFormat the observation with the predicted findings
        :param: predictedAnnotations, a DF containing all predicted annotations for a specific observation
        :return: a bool True if the observation contains only normal findings
        :return: the dataFrame with only key findings, if no key findings occurred, just return one line with None in the fields
        """
        keyFindings = predictedAnnotations[predictedAnnotations["HPO Term"].notnull() & predictedAnnotations["HPO Term"].str.startswith("HP:")]
        if len(keyFindings)==0:
            # if an observation has no key finding,
            normalFinding = pd.DataFrame([predictedAnnotations.iloc[0].copy()])
            normalFinding["HPO Term"] = np.nan
            normalFinding["Spans"] = np.nan
            return [True, normalFinding]
        else:
            #sanity check that we don't have incoherent annotations claiming the observation only having normal finding and also 1 or more key findings which contradict
            assert len(predictedAnnotations[predictedAnnotations["HPO Term"].isnull()])==0, f"I found incoherent predicted annotations for the observation [{predictedAnnotations['ObservationID']}], claiming that the observation only have normal finding and 1 or more key findings were found, check the predictions script."
            # one or multiple HPO terms are found
            return [False, keyFindings.copy()]


    def __addTNs(self, isOnlyTruthNormalFinding:bool, trueFindings:pd.DataFrame, isOnlyPredNormalFinding:bool, predFindings:pd.DataFrame):
        """
        Handle the case where: the observation does not contain any key findings, only normal findings
        and the system also predicted only normal findings
        :param: isOnlyTruthNormalFinding, a bool True if the observation only contain manually annotated Normal Findings
        :param: isOnlyPredNormalFinding, a bool True if the observation only contain predicted annotations Normal Findings
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: if we have TNs, we add them in the main alignment DF self.observations
        """
        if isOnlyTruthNormalFinding and isOnlyPredNormalFinding: #TNs
            self.TNs = self.TNs + 1
            if self.onlyNormalization:
                entry = {"ObservationID": trueFindings.iloc[0]["ObservationID"], "Text": trueFindings.iloc[0]["Text"], "Truth HPO ID": np.nan,  "Truth Spans": np.nan, "Truth Spans Text": np.nan, "Predicted HPO ID": np.nan, "Predicted Spans": np.nan, "Normalization": "TN"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = np.nan
                    entry["Predicted HPO Term"] = np.nan
                self.observations = self.observations.append(entry, ignore_index=True)
            else:
                self.eTNs = self.eTNs + 1
                self.oTNs = self.oTNs + 1
                entry = {"ObservationID": trueFindings.iloc[0]["ObservationID"], "Text": trueFindings.iloc[0]["Text"], "Truth HPO ID": np.nan, "Truth Spans":np.nan, "Truth Spans Text":np.nan, "Predicted HPO ID":np.nan, "Predicted Spans":np.nan, "Predicted Spans Text":np.nan, "Normalization":"TN", "Exact Extraction & Normalization": "TN", "Overlapping Extraction & Normalization": "TN"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = np.nan
                    entry["Predicted HPO Term"] = np.nan
                self.observations = self.observations.append(entry, ignore_index=True)


    def __addFPs(self, isOnlyTruthNormalFinding:bool, trueFindings:pd.DataFrame, isOnlyPredNormalFinding:bool, predFindings:pd.DataFrame):
        """
        Handle the case where: the observation does not contain any key findings, only normal findings,
        the system predicted (some) key findings, in that cas all predicted findings are FPs
        :param: isOnlyTruthNormalFinding, a bool True if the observation only contain manually annotated Normal Findings
        :param: isOnlyPredNormalFinding, a bool True if the observation only contain predicted annotations Normal Findings
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: if we have FPs, we add them in the main alignment DF self.observations
        """
        if isOnlyTruthNormalFinding and not isOnlyPredNormalFinding: #FPs
            for idx, pFind in predFindings.iterrows():
                self.FPs = self.FPs + 1
                if self.onlyNormalization:
                    entry = {"ObservationID": trueFindings.iloc[0]["ObservationID"], "Text": trueFindings.iloc[0]["Text"], "Truth HPO ID": np.nan, "Truth Spans": np.nan, "Truth Spans Text":np.nan, "Predicted HPO ID": pFind["HPO Term"], "Predicted Spans": pFind["Spans"], "Normalization": "FP"}
                    if self.hpo2term is not None:
                        entry["Truth HPO Term"] = np.nan
                        entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind["HPO Term"])
                    self.observations = self.observations.append(entry, ignore_index=True)
                else:
                    self.eFPs = self.eFPs + 1
                    self.oFPs = self.oFPs + 1
                    entry = {"ObservationID": trueFindings.iloc[0]["ObservationID"], "Text": trueFindings.iloc[0]["Text"], "Truth HPO ID": np.nan, "Truth Spans": np.nan, "Truth Spans Text":np.nan, "Predicted HPO ID": pFind["HPO Term"], "Predicted Spans": pFind["Spans"], "Predicted Spans Text": self.__extractSpansText(pFind["Text"], pFind["Spans"]), "Normalization": "FP", "Exact Extraction & Normalization": "FP", "Overlapping Extraction & Normalization": "FP"}
                    if self.hpo2term is not None:
                        entry["Truth HPO Term"] = np.nan
                        entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind["HPO Term"])
                    self.observations = self.observations.append(entry, ignore_index=True)


    def __addFNs(self, isOnlyTruthNormalFinding:bool, trueFindings:pd.DataFrame, isOnlyPredNormalFinding:bool, predFindings:pd.DataFrame):
        """
        Handle the case where: the observation discusses key findings,
        but the system predicted only normal findings, in that case all key findings labeled are missed (FNs)
        :param: isOnlyTruthNormalFinding, a bool True if the observation only contain manually annotated Normal Findings
        :param: isOnlyPredNormalFinding, a bool True if the observation only contain predicted annotations Normal Findings
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: if we have FNs, we add them in the main alignment DF self.observations
        """
        if isOnlyPredNormalFinding and not isOnlyTruthNormalFinding:
            for idx, tFind in trueFindings.iterrows():  # FNs
                self.FNs = self.FNs + 1
                if self.onlyNormalization:
                    entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text":self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": np.nan, "Predicted Spans": np.nan, "Normalization": "FN", }
                    if self.hpo2term is not None:
                        entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                        entry["Predicted HPO Term"] = np.nan
                    self.observations = self.observations.append(entry, ignore_index=True)
                else:
                    self.eFNs = self.eFNs + 1
                    self.oFNs = self.oFNs + 1
                    entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text":self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": np.nan, "Predicted Spans": np.nan, "Predicted Spans Text": np.nan, "Normalization": "FN", "Exact Extraction & Normalization": "FN", "Overlapping Extraction & Normalization": "FN"}
                    if self.hpo2term is not None:
                        entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                        entry["Predicted HPO Term"] = np.nan
                    self.observations = self.observations.append(entry, ignore_index=True)


    def __addPossibleTPs(self, isOnlyTruthNormalFinding: bool, trueFindings: pd.DataFrame, isOnlyPredNormalFinding: bool, predFindings: pd.DataFrame):
        """
        Handle the last case with possible TPs that we aligned - all findings not aligned are either FNs or FPs:
        the observation discusses key findings, the system also predicted key findings,
        we need to compare findings labeled and predicted
        :param: isOnlyTruthNormalFinding, a bool True if the observation only contain manually annotated Normal Findings
        :param: isOnlyPredNormalFinding, a bool True if the observation only contain predicted annotations Normal Findings
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted findings
        :return: We compare all annotations and find the TPs if any and add the remaining FPs/FNs, we add them in main alignment DF self.observations
        """
        if not isOnlyPredNormalFinding and not isOnlyTruthNormalFinding:
            if self.onlyNormalization:
                # for the normalization only the algorithm is simpler just consume the TP and handle the remaining errors
                self.__alignPossibleTPs(trueFindings, predFindings)
            else:
                # first align the exact matches, all will be TPs, eTPs, oTPs
                trueFindings, predFindings = self.__alignExactMatches(trueFindings, predFindings)
                # Then align the overlapping matches, all will be TPs, eFNs, oTPs
                trueFindings, predFindings = self.__alignOverlappingMatches(trueFindings, predFindings)
                # handle the remaining errors, some will be TPs eFNs oFNs or FNs, eFNs, oFNs or FPs, eFPs, oFPs
                self.__alignRemainingFindings(trueFindings, predFindings)

    def __alignPossibleTPs(self, trueFindings: pd.DataFrame, predFindings: pd.DataFrame):
        """
        If normalization only has been attempted (no extraction), we align the TPs and count the errors
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted findings
        """
        # all matches in this loop will be TPs
        idxTFMatched = []
        for idx, tFind in trueFindings.iterrows():# not ideal to loop in pandas, should be optimized. I left this for the moment
            #uncomment if you need to stop for debugging
            # if tFind["ObservationID"]=="0ef83261df80b0e500e2d4c7c79fb58c":#2017eff9a8feec4c751af86cdedec40f#32a92a682e3ac96c5810d74f1b9fee26
            #     log.debug("Stopped here.")
            pFind = predFindings[predFindings["HPO Term"] == tFind["HPO Term"]]
            if len(pFind)>0:
                self.TPs = self.TPs + 1
                pidxs = pFind.index.tolist()
                entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": pFind.loc[pidxs[0]]['HPO Term'], "Normalization": "TP"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                    entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind.loc[pidxs[0]]['HPO Term'])
                self.observations = self.observations.append(entry, ignore_index=True)
                predFindings.drop(labels=pidxs[0], axis=0, inplace=True)
                idxTFMatched.append(idx)
        trueFindings.drop(labels=idxTFMatched, axis=0, inplace=True)
        # All remaining in trueFindings are FNs since they did not match any predictions
        for idx, tFind in trueFindings.iterrows():
            self.FNs = self.FNs + 1
            entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": np.nan, "Normalization": "FN"}
            if self.hpo2term is not None:
                entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                entry["Predicted HPO Term"] = np.nan
            self.observations = self.observations.append(entry, ignore_index=True)
        # All remaining in the predFindings are FPs since they did not match any annotations
        for pidx, pf in predFindings.iterrows():
            self.FPs = self.FPs + 1
            entry = {"ObservationID": pf["ObservationID"], "Text": pf["Text"], "Truth HPO ID": np.nan, "Truth Spans": np.nan, "Truth Spans Text": np.nan, "Predicted HPO ID": pf['HPO Term'], "Normalization": "FP"}
            if self.hpo2term is not None:
                entry["Truth HPO Term"] = np.nan
                entry["Predicted HPO Term"] = self.__getPreferredTerm(pf['HPO Term'])
            self.observations = self.observations.append(entry, ignore_index=True)


    def __alignExactMatches(self, trueFindings: pd.DataFrame, predFindings: pd.DataFrame) -> list:
        """
        Search for the exact match if the extraction was attempted, the alignment is made on the positions of the disjoint spans
        the match is following the expected format (\d+-\d+,)*\d+-\d+ with the spans ordered as they occur in the text
        (ex. 1-6,17-25,29-37)
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: add the TPs, eTPs, oTPs in the main DF of the alignment self.observations
        """
        #keep a record of the index to delete them when a finding is matched
        idxTFMatched = []
        for idx, tFind in trueFindings.iterrows():# not ideal to loop in pandas but I do not see how to do it with paralellization
            #uncomment if you need to stop for debugging
            # if tFind["ObservationID"]=="2d891e09fd0ba9c5388711cafac921eb":
            #     log.debug("Stopped here.")
            pFind = predFindings[predFindings["HPO Term"] == tFind["HPO Term"]]
            for pidx, pf in pFind.iterrows():
                extracted = self.__compareFindingsPositions(tFind["Spans"], pf['Spans'])
                if extracted == "Exact": #strict match found
                    self.TPs  = self.TPs  + 1
                    self.eTPs = self.eTPs + 1
                    self.oTPs = self.oTPs + 1
                    #add the finding and keep the index for deletion
                    entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": pFind.loc[pidx]['HPO Term'], "Predicted Spans": pFind.loc[pidx]['Spans'], "Predicted Spans Text": self.__extractSpansText( pFind.loc[pidx]['Text'], pFind.loc[pidx]['Spans']), "Normalization": "TP", "Exact Extraction & Normalization": "TP", "Overlapping Extraction & Normalization": "TP"}
                    if self.hpo2term is not None:
                        entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                        entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind.loc[pidx]['HPO Term'])
                    self.observations = self.observations.append(entry, ignore_index=True)
                    idxTFMatched.append(idx)
                    predFindings.drop(labels=pidx, axis=0, inplace=True)
        trueFindings.drop(labels=idxTFMatched, axis=0, inplace=True)
        return [trueFindings, predFindings]


    def __alignOverlappingMatches(self, trueFindings: pd.DataFrame, predFindings: pd.DataFrame) -> list:
        """
        All exact matches have been deleted, we are now processing the possible findings labeled/predicted with overlapping spans
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: add the TPs, eFNs, oTPs in the main DF of the alignment self.observations
        """
        idxTFMatched = []
        for idx, tFind in trueFindings.iterrows():
            # if tFind["ObservationID"]=="2d891e09fd0ba9c5388711cafac921eb":
            #     log.debug("Stopped here.")
            bestPF = None
            bestPFidx = None
            pFind = predFindings[predFindings["HPO Term"] == tFind["HPO Term"]]
            if len(pFind)>0:
                for pidx, pf in pFind.iterrows():
                    extracted = self.__compareFindingsPositions(tFind["Spans"], pf['Spans'])
                    if extracted == "Overlapping": #Overlapping match found
                        if bestPF is None:
                            bestPF = pf
                            bestPFidx = pidx
                        else:
                            bestPF, bestPFidx = self.__getBestOverlappingFinding(tFind, bestPF, bestPFidx, pf, pidx)
            if bestPF is not None:
                self.TPs  = self.TPs  + 1
                self.eFNs = self.eFNs + 1 # missed the exact match for this term
                self.oTPs = self.oTPs + 1 # but found the overlapping
                #add the finding and keep the index for deletion
                entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": bestPF['HPO Term'], "Predicted Spans": bestPF['Spans'], "Predicted Spans Text": self.__extractSpansText(bestPF['Text'], bestPF['Spans']), "Normalization": "TP", "Exact Extraction & Normalization": "FN", "Overlapping Extraction & Normalization": "TP"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                    entry["Predicted HPO Term"] = self.__getPreferredTerm(bestPF['HPO Term'])
                self.observations = self.observations.append(entry, ignore_index=True)
                idxTFMatched.append(idx)
                predFindings.drop(labels=bestPFidx, axis=0, inplace=True)
        trueFindings.drop(labels=idxTFMatched, axis=0, inplace=True)
        return [trueFindings, predFindings]


    def __alignRemainingFindings(self,trueFindings: pd.DataFrame, predFindings: pd.DataFrame):
        """
        All exact matches and overlapping have been processed the best we could, remaining Findings that do not overlap
        are FPs/FNs
        :param: trueFindings, the DF with all manually annotated findings
        :param: predFindings, the DF with all predicted annotations in findings
        :return: add possible TPs, and other errors in the main DF of the alignment self.observations
        """
        for idx, tFind in trueFindings.iterrows():
            pFind = predFindings[predFindings["HPO Term"] == tFind["HPO Term"]]
            if len(pFind) > 0:  # TP
                self.TPs = self.TPs + 1
                self.eFNs = self.eFNs + 1 #FN since the term have been found but not at this position so missed
                self.oFNs = self.oFNs + 1 #FN idem, not even a span in common
                entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": pFind.iloc[0]['HPO Term'], "Predicted Spans": pFind.iloc[0]['Spans'], "Predicted Spans Text": self.__extractSpansText(pFind.iloc[0]['Text'], pFind.iloc[0]['Spans']), "Normalization": "TP", "Exact Extraction & Normalization": "FN", "Overlapping Extraction & Normalization": "FN"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                    entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind.iloc[0]['HPO Term'])
                self.observations = self.observations.append(entry, ignore_index=True)
                # remove the prediction since it has been aligned with a truth key finding (if there are multiple mention of the same Term, we just remove the first mention)
                predFindings.drop(labels=pFind.index[0], axis=0, inplace=True)
            else:  # FN
                self.FNs = self.FNs + 1
                self.eFNs = self.eFNs + 1
                self.oFNs = self.oFNs + 1
                entry = {"ObservationID": tFind["ObservationID"], "Text": tFind["Text"], "Truth HPO ID": tFind["HPO Term"], "Truth Spans": tFind["Spans"], "Truth Spans Text": self.__extractSpansText(tFind["Text"], tFind["Spans"]), "Predicted HPO ID": np.nan, "Predicted Spans": np.nan, "Predicted Spans Text": np.nan, "Normalization": "FN", "Exact Extraction & Normalization": "FN", "Overlapping Extraction & Normalization": "FN"}
                if self.hpo2term is not None:
                    entry["Truth HPO Term"] = self.__getPreferredTerm(tFind["HPO Term"])
                    entry["Predicted HPO Term"] = np.nan
                self.observations = self.observations.append(entry, ignore_index=True)
            # we 'consumed' all truth key findings, all remaining predicted key findings are FPs...
        for pidx, pFind in predFindings.iterrows():
            self.FPs = self.FPs + 1
            self.eFPs = self.eFPs + 1
            self.oFPs = self.oFPs + 1
            entry = {"ObservationID": pFind["ObservationID"], "Text": pFind["Text"], "Truth HPO ID": np.nan, "Truth Spans": np.nan, "Truth Spans Text": np.nan, "Predicted HPO ID": pFind["HPO Term"], "Predicted Spans": pFind["Spans"], "Predicted Spans Text": self.__extractSpansText(pFind["Text"], pFind["Spans"]), "Normalization": "FP", "Exact Extraction & Normalization": "FP", "Overlapping Extraction & Normalization": "FP"}
            if self.hpo2term is not None:
                entry["Truth HPO Term"] = np.nan
                entry["Predicted HPO Term"] = self.__getPreferredTerm(pFind["HPO Term"])
            self.observations = self.observations.append(entry, ignore_index=True)


    def __extractSpansText(self, text: str, spans:str) -> str:
        """
        Extract and return the text of a valid sequence of the given spans
        :param: text, the full text of an observation
        :param: spans, a valid sequence of spans anchoring the mention of the Term
        :return: the spans denoted by the spans (' - ' separate the texts of disjoint spans)
        """
        txtExt = []
        sps = spans.split(",")
        for span in sps:
            positions = span.split("-")
            txtExt.append(text[int(positions[0]): int(positions[1])])
        txtExt = ' - '.join(txtExt)
        return txtExt


    def __getBestOverlappingFinding(self, truthFinding:pd.Series, predFind1:pd.Series, predFind1Idx:int, predFind2:pd.Series, predFind2Idx:int) -> list:
        """
        When extraction is attempted, we found 2 predicted findings that overlap with the truth Finding,
        we need to find the one which shares most characters with the truth Finding
        :param: truthFinding the truth finding matched by the 2 findings predicted
        :param: predFind1, the first finding predicted that overlap with truthFinding, this is currently the best candidate
        :param: predFind1Idx, the index of the first finding predicted
        :param: predFind2, the second finding predicted that overlap with truthFinding, this is a new candidate
        :param: predFind2Idx, the index of the second finding predicted
        :return: the serie and index of the finding which overlap the most with the truth finding
        """
        def __getOverlappingSize(spans1, spans2)->int:
            overlap = 0
            for span1 in spans1:
                for span2 in spans2:
                    if not ((span1[1] < span2[0]) or (span2[1] < span1[0])):#they overlap, but how much
                        if   span1[0]<span2[0] and span2[0]<=span1[1] and span1[1]<span2[1]:
                            overlap = overlap + ((span1[1]-span2[0])+1)# the only part overlapping between the finding on this span
                        elif span2[0]<=span1[0] and span1[1]<=span2[1]: #exact overlap is allowed since another span maybe different
                            overlap = overlap + ((span1[1]-span1[0])+1)# span 1 is nested in span 2, so the overlap is the length of span 1
                        elif span1[0]<=span2[0] and span2[1]<=span1[1]: #exact overlap is allowed since another span maybe different
                            overlap = overlap + ((span2[1] - span2[0])+1)  # span 2 is nested in span 1, so the overlap is the length of span 2
                        elif span2[0]<span1[0] and span1[0]<=span2[1] and span2[1]<span1[1]:
                            overlap = overlap + ((span2[1] - span1[0])+1)  # the only part overlapping between the finding on this span
                        else:
                            raise Exception(f"I was not expecting this type of overlapping between two findings, check code and inputs: truth Finding -> {spans1}, predicted finding -> {spans2}")
            return overlap

        def testOverlapping():
            """
            Just for testing the function __getOverlappingSize if need be...
            """
            posTruth = self.__getPositions("1-6")  #" 111111"
            posPred1 = self.__getPositions("6-8")  #"      222"
            assert __getOverlappingSize(posTruth, posPred1) == 1, "Ooop 1..."
            posTruth = self.__getPositions("1-6")  #" 111111"
            posPred1 = self.__getPositions("4-8")  #"    22222"
            assert __getOverlappingSize(posTruth, posPred1) == 3, "Ooop 2..."
            posTruth = self.__getPositions("1-4")  #" 1111"
            posPred1 = self.__getPositions("1-6")  #" 222222"
            assert __getOverlappingSize(posTruth, posPred1) == 4, "Ooop 3..."
            posTruth = self.__getPositions("2-5")  # 1111
            posPred1 = self.__getPositions("1-6")  # 222222
            assert __getOverlappingSize(posTruth, posPred1) == 4, "Ooop 4..."
            posTruth = self.__getPositions("2-6")  # 11111
            posPred1 = self.__getPositions("1-6")  # 222222
            assert __getOverlappingSize(posTruth, posPred1) == 5, "Ooop 5..."
            posTruth = self.__getPositions("2-8")  #"  1111111"
            posPred1 = self.__getPositions("1-5")  #" 22222"
            assert __getOverlappingSize(posTruth, posPred1) == 4, "Ooop 6..."
            posTruth = self.__getPositions("5-8")  #"     1111"
            posPred1 = self.__getPositions("2-5")  #"  2222"
            assert __getOverlappingSize(posTruth, posPred1) == 1, "Ooop 7..."
            posTruth = self.__getPositions("1-6")  # 111111
            posPred1 = self.__getPositions("1-4")  # 2222
            assert __getOverlappingSize(posTruth, posPred1) == 4, "Ooop 8..."
            posTruth = self.__getPositions("1-6")  # 111111
            posPred1 = self.__getPositions("2-5")  # 2222
            assert __getOverlappingSize(posTruth, posPred1) == 4, "Ooop 9..."
            posTruth = self.__getPositions("1-6")  # 111111
            posPred1 = self.__getPositions("2-6")  # 22222
            assert __getOverlappingSize(posTruth, posPred1) == 5, "Ooop 10..."
            posTruth = self.__getPositions("1-6,9-11")  # 111111  111
            posPred1 = self.__getPositions("2-6")  # 22222
            assert __getOverlappingSize(posTruth, posPred1) == 5, "Ooop 11..."
            posTruth = self.__getPositions("2-6")  # 22222
            posPred1 = self.__getPositions("1-6,8-11")  # 111111 1111
            assert __getOverlappingSize(posTruth, posPred1) == 5, "Ooop 12..."
            posTruth = self.__getPositions("2-8,13-17")  #"  1111111    11111"
            posPred1 = self.__getPositions("1-5,12-15")  #" 22222      2222"
            assert __getOverlappingSize(posTruth, posPred1) == 7, "Ooop 13..."
            posTruth = self.__getPositions("2-8,10-11,13-18,22-26")  #"  1111111 11 111111   11111"
            posPred1 = self.__getPositions("1-5,12-15,22-25")        #" 22222      2222      2222
            assert __getOverlappingSize(posTruth, posPred1) == 11, "Ooop 14..."
        # testOverlapping()
        posTruth = self.__getPositions(truthFinding["Spans"])
        posPred1 = self.__getPositions(predFind1["Spans"])
        truthPred1Over = __getOverlappingSize(posTruth, posPred1)
        posPred2 = self.__getPositions(predFind2["Spans"])
        truthPred2Over = __getOverlappingSize(posTruth, posPred2)
        if truthPred1Over<=truthPred2Over:
            return [predFind1, predFind1Idx]
        else:
            return [predFind2, predFind2Idx]


    def __compareFindingsPositions(self, spansTruth:str, spansPred:str) -> str:
        """
        Compare the extraction given by the annotator and predicted by the system if the system attempted to extract HPO terms
        :param: spansTruth, the spans labeled by the annotator, as a string following the expected format
        :param: spansPred, the spans predicted by the system, as a string following the expected format
        :return: str: 'Exact', exact match between the spans; 'Overlapping', overlapping but not exact match; 'Disjoint', not overlapping at all; 'NA' not extracted (NA)
        """
        if pd.isnull(spansPred) or len(spansPred)==0:
            return "NA"
        if spansPred==spansTruth:
            return "Exact"
        # not empty, not strict match need to compare the spans sequences
        posTruth = self.__getPositions(spansTruth)
        posPred = self.__getPositions(spansPred)
        for posP in posPred:
            for posT in posTruth:
                if not ((posT[1]<posP[0]) or (posP[1]<posT[0])): #if not (either the Truth span occurs before or Truth occurs after), then they overlap
                    return "Overlapping"
        return "Disjoint"# could not find an overlapping span between the two Terms


    def __getPositions(self, spansFinding:str) -> list:
        """
        read the positions of the finding in the valid Format, return the list for further processing
        :param: spansFinding, a list of spans following our format
        :return: the list of lists of int, eg: 1-4,12-17,34-38 -> [[1,4],[12,17],[34,38]]
        """
        spans = spansFinding.split(",")
        positions = []
        for span in spans:
            pos = span.split("-")
            positions.append([int(pos[0]), int(pos[1])])
        return positions


    def getAlignedFindings(self) -> pd.DataFrame:
        """
        :return: the DataFrame with all annotations (manually annotated and predicted) aligned for verification
        """
        return self.observations


    def getF1Scores(self) -> dict:
        """
        Compute the F1 score for the normalization with the alignment performed,
        if spans were available, then it also computes the F1 scores for exact + overlapping extraction & normalization
        :return: dict: {Normalization_F1, Normalization_Precision, Normalization_Recall}
                        optional keys: [exactExtNorm_F1, exactExtNorm_Precision, exactExtNorm_Recall,
                                        overExtNorm_F1, overExtNorm_Precision, overExtNorm_Recall]
        """
        def _f1(Prec:float, Rec:float)->float:
            if Prec==0.0 and Rec==0.0:
                log.error("Precision and Recall are both 0.0, division by 0, F1 is set to 0.0")
                F1 = 0.0
            else:
                F1 = 2*(Prec*Rec)/(Prec+Rec)
            return F1
        #Normalization only
        Prec = self.TPs/(self.TPs+self.FPs)
        Rec =  self.TPs/(self.TPs+self.FNs)
        F1 = _f1(Prec, Rec)
        scores = {"Normalization_F1":F1, "Normalization_Precision":Prec, "Normalization_Recall":Rec}

        if not self.onlyNormalization:
            #exact extraction + Normalization
            if (self.eTPs+self.eFPs)>0: ePrec = self.eTPs/(self.eTPs+self.eFPs)
            else: ePrec = 0.0
            if (self.eTPs+self.eFNs)>0: eRec = self.eTPs/(self.eTPs+self.eFNs)
            else: eRec = 0.0
            eF1 = _f1(ePrec, eRec)
            scores.update({"exactExtNorm_F1": eF1, "exactExtNorm_Precision": ePrec, "exactExtNorm_Recall": eRec})

            #overlapping extraction + Normalization
            if (self.oTPs+self.oFPs)>0: oPrec = self.oTPs/(self.oTPs+self.oFPs)
            else: oPrec = 0.0
            if (self.oTPs+self.oFNs)>0: oRec = self.oTPs/(self.oTPs+self.oFNs)
            else: oRec = 0.0
            oF1 = _f1(oPrec, oRec)
            scores.update({"overExtNorm_F1": oF1,  "overExtNorm_Precision": oPrec,  "overExtNorm_Recall": oRec})

        return scores


class HPOEvaluator(object):
    def __init__(self):
        """
        Evaluator for the BioCreative Shared tasks Track 3: Genetic Phenotype Extraction and Normalization from
        Dysmorphology Physical Examination Entries
        """
        return


    def evaluate(self, pathTruth:str, pathPreds:str):
        """
        Given a gold standard and a set of predictions on this gold standard, compute the findings which holds the F1/Precision/Recall metrics
        :param: pathTruth, a valid path to the .tsv following the format defined in https://biocreative.bioinformatics.udel.edu/news/biocreative-viii/track-3/
        :param: pathPreds, a valid path to the .tsv following the format defined in https://biocreative.bioinformatics.udel.edu/news/biocreative-viii/track-3/
        :return: the Findings holding the scores
        """
        truth = self.__readAnns(pathTruth, False)
        pred = self.__readAnns(pathPreds, True)
        obsInTruth = truth["ObservationID"].unique().tolist()
        obsInTruth.sort()
        def removeDecoys(truth, pred) -> pd.DataFrame:
            """
            The unlabeled test set may contain decoys added for the BioCreative competition, we remove them by looking at the Truth
            :param truth: the manually annotated test set
            :param pred: the predictions made on the test set and the decoys
            :return: a df with only the predictions made on the observations of the test set
            """
            decoys = pred[~pred["ObservationID"].isin(truth["ObservationID"].unique())].index
            if len(decoys)>0:
                pred.drop(decoys, inplace=True)
            return pred

        pred = removeDecoys(truth, pred)
        obsInPred = pred["ObservationID"].unique().tolist()
        obsInPred.sort()
        assert ((len(obsInTruth) == len(obsInPred)) and (obsInPred == obsInTruth)), f"The set of observations IDs in the Truth is not the same than in the Prediction, where it should. Please check the input files, Truth:{pathTruth}; Preds:{pathPreds}."

        findings = Findings(truth, pred)
        return findings


    def __readAnns(self, pathAnns: str, isPrediction:bool) -> pd.DataFrame:
        """
        Read the annotation from the path given and validate the inputs for further processing
        :param: pathAnns, a valid path to observations with either truth or predicted annotations
        :param: isPrediction, true if the file read the prediction file, false if the file read is the truth file
        :return: the dataframe representing the annotations validated
        """
        anns = pd.read_csv(pathAnns, sep="\t")
        self.__validateInputs(anns, pathAnns, isPrediction)
        return anns


    def __validateInputs(self, anns: pd.DataFrame, pathAnns:str, isPrediction:bool):
        """
        Check if the annotations of the observations given in inputs are valid
        (check columns names, duplicates entries, inconsistency in the annotations)
        :param: anns, a df with the annotations read from a valid tsv
        :return: raise an exception if anything is wrong with the inputs
        """
        log.info(f"Checking the inputs for {pathAnns}...")
        assert "ObservationID" in anns.columns, f"I could not find the column ObservationID in the tsv file read@{pathAnns}"
        assert "Text" in anns.columns, f"I could not find the column Text in the tsv file read@{pathAnns}"
        assert "HPO Term" in anns.columns, f"I could not find the column HPO Term in the tsv file read@{pathAnns}"
        assert len(anns[anns["HPO Term"].isnull() | anns["HPO Term"].str.startswith("HP:")]) == len(anns), f"I found predicted annotations for which the HPO Term is not NA and that do not start with 'HPO Term', check the input format."
        if isPrediction: #prediction file
            assert "Polarity" not in anns.columns, f"I find the column Polarity in the file of the prediction, participants should not predict the negated concepts, please remove the column Polarity and the predictions @{pathAnns}"
        else: #truth file
            assert "Polarity" in anns.columns, f"I could not find the column Polarity in the truth file, check the input file @{pathAnns}"
            # in the gold standard, check that all negated terms are not also affirmed in an observation, should reject:
            # 000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:0000218	NA	14-32
            # 000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:0000218	X	14-32
            dups = anns.duplicated(subset=["ObservationID", "HPO Term", "Spans"], keep="first")
            assert not dups.any(), f"I found HPO Terms affirmed to be mentioned in the observation negated in a different line, check the inputs @{pathAnns}: {anns[anns.duplicated(subset=['ObservationID', 'HPO Term', 'Spans'], keep='first')]['ObservationID'].tolist()}"

        if "Spans" not in anns.columns:
            anns["Spans"] = np.nan
        #check that an observation claimed having a key finding is not also claimed having no key finding, should reject:
        #000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:0000218	NA	14-32
        #000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	NA	NA	NA
        obsWKeys = set(anns[anns['HPO Term'].notnull() & anns['HPO Term'].str.startswith("HP:")]["ObservationID"].tolist())
        obsWNorms = set(anns[anns['HPO Term'].isnull()]["ObservationID"].tolist())
        assert len(obsWKeys.intersection(obsWNorms))==0, f"I found observations annotated with key findings and annotated as only normal findings, check the inputs @{pathAnns}: {obsWKeys.intersection(obsWNorms)}"

        if anns["Spans"].notnull().any():# Spans are given for some observations, so computing performance on extraction is possible
            # check if there is any duplicated line (all fields)
            assert not anns.duplicated(keep="first").any(), f"I found duplicates row(s) in the input data, check the inputs @{pathAnns}: {anns[anns.duplicated(keep='first')]['ObservationID'].tolist()}"
            #if a normalization is given the spans should be given, should reject
            #000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:002	NA	NA
            assert len(anns[anns["HPO Term"].notnull() & anns["Spans"].isnull()])==0, f'I found Term normalized but not extracted whereas terms in other observations have been extracted, check the inputs @{pathAnns}: {anns[anns["HPO Term"].notnull() & anns["Spans"].isnull()]["ObservationID"].tolist()}'
            #if the spans are given, check that all mentions are anchored, should reject:
            #000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	NA	NA	34-40,50-56
            assert len(anns[anns["HPO Term"].isnull() & anns["Spans"].notnull()])==0, f'I found spans extracted but not normalized, check the inputs @{pathAnns}: {anns[anns["HPO Term"].isnull() & anns["Spans"].notnull()]["ObservationID"].tolist()}'
            #check that all spans are formated and within the length of the text
            # format of the spans: (\d+-\d+,)*\d+-\d+
            def checkSpans(finding:pd.Series):
                if ((not pd.isnull(finding["Spans"])) and len(finding["Spans"]))>0:
                    spans = finding["Spans"].split(",")
                    assert len(spans)>0, f'I found a Spans value that I cannot process check the inputs @{pathAnns}: {finding["Spans"]}'
                    lastPosition = -1
                    for span in spans:
                        beginend = span.split("-")
                        assert (len(beginend)==2 and beginend[0].isdigit() and beginend[1].isdigit()), f'I found a list of spans incorrectly formated, should be (\d+-\d+,)*\d+-\d+, check the inputs @{pathAnns}: {finding["ObservationID"]} -> {finding["Spans"]}'
                        # I reject annotation of 1 character, should never happen...
                        assert int(beginend[0])<int(beginend[1]) and 0<=int(beginend[0]) and int(beginend[1])<=len(finding["Text"]), f'I found a span with positions that are incoherent/out of range given the Text of the observation, check the inputs @{pathAnns}: {finding["ObservationID"]} -> {finding["Spans"]}, length text: {len(finding["Text"])}'
                        # all spans should be mutually exclusive within a Term, not overlapping, and ordered
                        assert lastPosition < int(beginend[0]), f'I found a term with an invalid spans sequence, i.e. overlapping spans or not ordered spans, check the inputs @{pathAnns}: {finding["ObservationID"]} -> [{finding["Spans"]}], length text: {len(finding["Text"])}'
                        lastPosition = int(beginend[1])
            anns.apply(lambda finding: checkSpans(finding), axis=1)
            #check if a span is exactly covered by two or more terms with different HPO IDs, should reject:
            # 000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:0000218	NA	14-32
            # 000f780da593b746a7cc4753de22a2ce	FAD94BB3B7D6DFF	MOUTH: Mildly high arched palate. Normal lips and tongue.	HP:0003000	NA	14-32
            dups = anns.duplicated(subset=["ObservationID", "Spans"], keep="first")
            assert not dups.any(), f"I found the same spans labeled with two or more HPO Terms in these observations, this is not supported. Check the inputs @{pathAnns}: {anns[anns.duplicated(subset=['ObservationID', 'Spans'], keep='first')]['ObservationID'].tolist()}"
            log.info("Everything looks good.")

def testscript():
    """
    Run the script over fake data to test the evaluation script
    :return: raise exception if the script does not compute the scores expected
    """
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler("/Users/weissenbacd/Documents/tmp/HPOEval.log", 'w', 'utf-8')])
    # The dev set was used for testing the script, here are the errors introduced and that should be detected by the evaluation script:
    #     - 0604f751275118b36c6ba978121d0c32: 1 TP (1 eTP, 1 oTP)
    #     - 000f780da593b746a7cc4753de22a2ce: 2 TPs (2 eTPs, 2 oTPS)
    #     - 001d6c29f4e6ab6d37e2a4b0b84db25c: 1 TP, 1 FP (1 eTP, 1 oTP; 1 eFP, oFP)
    #     - 063793e003f1cfc8ac17d23a9e18ed17: 2 TPs, 2 FPs (2 eTP, 2 oTP, 2oTP 2 oTP)
    #     - 063ca6b80f43f44e9e634313ecf30474: 1 FP (1 eFP, 1 oFP)
    #     - 018fa5440489fec2f79b944c77c14a2b: 2 FPs, (2 eFP, 2 oFP)
    #     - 06835d763de6a26b9886de9bfcc66d5f: 2 FPs (with 1 normal findings to ignore in dev set) (2 eFP, 2 oFP)
    #     - 05e04caed7bea08c9dc112c4a9af2082: 4 FPs, (with all 4 normnal findings to ignore in dev set) (4 eFPs, 4 oFPs)
    #     - 028ec43d4e6aabeac24997ad73c4b4e0: 1 TN (1 eFN, 2oTN)
    #     - 050343224a11ccf9d1765af454194306: 2 FNs, (with 1 normal findings to ignore in dev set) (2 eFP, 2 oFP)
    #     - 0689279eb59f1a8ce7b8e70dc342c17b: 1 TN, (with 1 normal finding to ignore) (1 eTN, 1 oTN)
    #     - 0691bad7f6cddd0603f23d3dc595bbf4: 1 TN, (with 2 normal findings to ignore) (1 eTN 1 oTN)
    #     - 078430ce52d08b4f2fed07769432b2fb: 2 TPs, 2 FPs, (with 2 normal findings to ignore) (2 eTP, 2 oTP, 2 eFP 2 oFP)
    #     - 0804df2112a91ec2965d2af4af9fe510: 3 TPs, 3 FPs, 2 FNs, (with 2 normal findings to ignore) (3 eTP, 3 oTP, 2 eFN, 2 oFN, 2 eFP, 2 oFP)
    #     - 08249078b3a5ec30cde73179a90eef35: 4 TPs, 4 eTP, 4 oTP
    #     - 1b3e4d42d5683e8a9e34f96db9346d52: 3 FNs, with 3 mentions of the same Term in Truth (3 eFN, 3 oFN)
    #     - 1d681e4b4d41502056757938e670f8a4: 3 FPs, with 3 mentions of the same Term in Preds (3 eFP, 3 oFP)
    #     - 2011c130bb115d37613d486ce901c55f: 1 TP, 2 FN, with 3 mentions of the same Term in Truth exact match third mention (1 eTP, 1 oTP, 2 eFN 2 oFN)
    #     - 206a333c81539fd5a9026a65852010ca: 1 TP, 2 FN, with 3 mentions of the same Term in Truth exact match first mention (1 eTP, 1 oTP, 2 eFN, 2 oFN)
    #     - 20dbff9744cb1342682a933a8242ea18: 1 TP, 2 FP, with 3 mentions of the same Term in Pred exact match second mention (1 eTP,  oTP, 2 eFP, 2 oFP)
    #     - 2174230cc71fea22297197c68fa38219: 1 TP, 1 FN, with 2 mentions of the same term in Truth and Pred with overlap for the terms in Pred, 30-36 is the longer and should be selected (1 eFN 1 oTP, 1 eFN 1oFN)
    #     - 236792c4eaa86ec8f4a28d7203d3ed24: 3 TP, 4 FN, 1 FP with overlapping (1 eTP 1 oTP, 2 eFN 2 oTP, 4 eFN, 4 oFN, 1 eFP 1 oFP)
    #     - 033f7020c6330defb4a15d018bf8634e: 1 TP (1 eTP, 1 oTP)
    #     - 0345bde9f1b49a75228256f8d45ecd54: 1 TP (1 eTP, 1 oTP)
    #     - 039fd9cc8b369630224720c622305d57: 1 TP, 1FP (1eTP 1oTP, 1 eFP 1oFP)
    #     - 048e125ed3e80bdd0fe9158e84686c88: 1 TP (1 eTP, 1 oTP)

    # Performance to be computed on the case test:
    # => 17:FN, 21:FP, 3:TN, 27:TP, 'Normalization_F1': 0.5869565217391304, 'Normalization_Precision': 0.5625, 'Normalization_Recall': 0.6136363636363636, 'Normalization_Confusion': 'TPs:27, TNs:3, FPs:21, FNs:17',
    # => 21:eFN, 21:eFP, 3:TN, 23:eTP
    # => 18:oFN, 21:oFP, 3:TN, 26:oTP

    PATH_TEST_DATA = str(Path(os.path.abspath(os.curdir)).parent)+"/data/"

    #First test extraction and Normalization
    log.debug("Running internal test of the script...")
    PATH_TRUTH = PATH_TEST_DATA+"PhenormTestEvaluationScriptDataTruth.tsv"
    PATH_PREDICTIONS = PATH_TEST_DATA+"PhenormTestEvaluationScriptDataPrediction.tsv"
    eval = HPOEvaluator()
    findings = eval.evaluate(PATH_TRUTH, PATH_PREDICTIONS)
    findings.getAlignedFindings().to_csv("/tmp/alignmentPheNormTestEvaluationScript.tsv", sep="\t", index=False)
    normConf = f"TPs:{findings.TPs}, TNs:{findings.TNs}, FPs:{findings.FPs}, FNs:{findings.FNs}"
    assert (findings.getF1Scores()['Normalization_F1'] == 0.5869565217391304 and findings.getF1Scores()['Normalization_Precision'] == 0.5625 and findings.getF1Scores()['Normalization_Recall'] == 0.6136363636363636), f"I did not find the expected score for the unit test cases, check the code."
    assert normConf =='TPs:27, TNs:3, FPs:21, FNs:17', f"I did not find the expected score for the unit test cases, check the code."
    exConf = f"TPs:{findings.eTPs}, TNs:{findings.eTNs}, FPs:{findings.eFPs}, FNs:{findings.eFNs}"
    ovConf = f"TPs:{findings.oTPs}, TNs:{findings.oTNs}, FPs:{findings.oFPs}, FNs:{findings.oFNs}"
    assert (findings.getF1Scores()['exactExtNorm_F1']==0.5227272727272727 and findings.getF1Scores()['exactExtNorm_Precision']==0.5227272727272727 and findings.getF1Scores()['exactExtNorm_Recall']==0.5227272727272727), f"I did not find the expected score for the unit test cases, check the code."
    assert exConf=='TPs:23, TNs:3, FPs:21, FNs:21', f"I did not find the expected score for the unit test cases, check the code."
    assert (findings.getF1Scores()['overExtNorm_F1']==0.5714285714285714 and findings.getF1Scores()['overExtNorm_Precision']==0.5531914893617021 and findings.getF1Scores()['overExtNorm_Recall']==0.5909090909090909), f"I did not find the expected score for the unit test cases, check the code."
    assert ovConf=='TPs:26, TNs:3, FPs:21, FNs:18', f"I did not find the expected score for the unit test cases, check the code."

    #Then test normalization only
    predWOSpans = pd.read_csv(PATH_PREDICTIONS, sep="\t")
    predWOSpans.drop(["Spans"], axis=1, inplace=True)
    PATH_PREDICTIONS = "/tmp/PheNormTestDataWOSpans.tsv"
    predWOSpans.to_csv(PATH_PREDICTIONS, sep="\t", index=False)
    eval = HPOEvaluator()
    findings = eval.evaluate(PATH_TRUTH, PATH_PREDICTIONS)
    findings.getAlignedFindings().to_csv("/tmp/alignmentPheNormTestEvaluationScriptWOSpan.tsv", sep="\t", index=False)
    normConf = f"TPs:{findings.TPs}, TNs:{findings.TNs}, FPs:{findings.FPs}, FNs:{findings.FNs}"
    assert (findings.getF1Scores()['Normalization_F1'] == 0.5869565217391304 and findings.getF1Scores()['Normalization_Precision'] == 0.5625 and findings.getF1Scores()['Normalization_Recall'] == 0.6136363636363636), f"I did not find the expected score for the unit test cases, check the code."
    assert normConf =='TPs:27, TNs:3, FPs:21, FNs:17', f"I did not find the expected score for the unit test cases, check the code."
    log.debug("Test successful.")


def evaluate():
    """
    Run locally the evaluation script on various systems' predictions
    :return: the scores of those systems
    """
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler("/Users/weissenbacd/Documents/tmp/HPOEval.log", 'w', 'utf-8')])

    def eval(PATH_TRUTH:str, PATH_PREDICTIONS:str, PATH_ALIGNMENT:str):
        eval = HPOEvaluator()
        findings = eval.evaluate(PATH_TRUTH, PATH_PREDICTIONS)
        log.info(findings.getF1Scores())
        findings.getAlignedFindings().to_csv(PATH_ALIGNMENT, sep="\t", index=False)

    PATH_TRUTH = "/Users/weissenbacd/tal/Ecrit/Workshops/Accepted/BioCreativeVIII23/Data/TEST_SET/TEST_SET_WITH_SOLUTION/BioCreativeVIII3_TestSet.tsv"

    log.info("Start evaluating Doc2HPO system...")
    PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/Baselines/doc2hpo_prediction_log.tsv"
    PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentDoc2HPO.tsv"
    eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)

    # log.info("Start evaluating NCR system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/Baselines/NCR_prediction_log.tsv"
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentNCR.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)

    # log.info("Start evaluating PhenoBERT system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/Baselines/phenobert_prediction_log.tsv"
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentPhenoBERT.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)

    # log.info("Start evaluating PhenoTagger system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/Baselines/phenotagger_prediction_log.tsv"
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentPhenoTagger.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)
    #
    # log.info("Start evaluating Txt2HPO system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/Baselines/txt2hpo_prediction_log.tsv"
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentTxt2HPO.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)
    #
    # log.info("Start evaluating Experiment1 system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/exp1-normalizer-test-preds-biocreative.tsv"
    # # Issue with some inputs: 0e3b133d45b4460a071e3871c9427f53 missing in the predictions...
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmenExperiment1.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)
    #
    # log.info("Start evaluating Experiment2 system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/exp2-normalizer-test-preds-biocreative.tsv"
    # # Issue with some inputs: 0e3b133d45b4460a071e3871c9427f53 missing in the predictions...
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmenExperiment2.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)
    #
    # log.info("Start evaluating Experiment3 system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/exp3-normalizer-test-preds-biocreative.tsv"
    # # Issue with some inputs: 0e3b133d45b4460a071e3871c9427f53 missing in the predictions...
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmenExperiment3.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)
    #
    # log.info("Start evaluating Experiment4 system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/tal/Soft_Developement/Ressources/Dev_Genetics/Experiments/exp4-normalizer-test-preds-biocreative.tsv"
    # # Issue with some inputs: 0e3b133d45b4460a071e3871c9427f53 missing in the predictions...
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmenExperiment4.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)

    # log.info("3a. Start evaluating API output system...")
    # PATH_PREDICTIONS = "/Users/weissenbacd/Documents/tmp/Phenorm_api_results.tsv"
    # # Issue with some inputs: 0e3b133d45b4460a071e3871c9427f53 missing in the predictions...
    # PATH_ALIGNMENT = "/Users/weissenbacd/Documents/tmp/alignmentPheNormAPI3a.tsv"
    # eval(PATH_TRUTH, PATH_PREDICTIONS, PATH_ALIGNMENT)


def evaluateBioCreative():
    """
    The MAIN FUNCTION to run to evaluate the predictions made during the BioCreative VIII Task 3
    This is the function called by Codalab
    :param: the task evaluated '3a'/'3b', 3a will perfom normalization only, 3b extraction & normalization
    :param: 'training'/'evaluation', training will run the evaluation on the development set, evaluation on the test set
    """
    # First, check where the .py is running (codalab or locally) and gather the parameters...
    # as per the metadata file, input, output directories are the arguments and subtask a parameter of the evaluation script
    if len(sys.argv)==5: #call from codalab
        log.basicConfig(level=log.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[log.StreamHandler(sys.stdout)])
        log.warning("Start scoring BioCreative  VIII Task 3...")
        print("Start scoring BioCreative  VIII Task 3...")
        [_, input_dir, output_dir, subtask, period] = sys.argv
        # unzipped reference data is always in the 'ref' subdirectory
        # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
        if period == "training":
            PATH_TRUTH = os.path.join(input_dir, 'ref', 'BioCreativeVIII3_ValSet.tsv')
        elif period == "evaluation":
            PATH_TRUTH = os.path.join(input_dir, 'ref', 'BioCreativeVIII3_TestSet.tsv')
        else:
            msg = f"Unknown period given as parameter in the metadata file, check the setting: {period}"
            log.fatal(msg)
            raise Exception(msg)
        # unzipped submission data is always in the 'res' subdirectory
        # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
        PATH_PREDICTIONS = os.path.join(input_dir, 'res', 'prediction_BioCreativeVIIITask3.tsv')
        if not os.path.exists(PATH_PREDICTIONS):
            msg = f'I could not find the file of predictionS as expected. Check the name of the file submitted, it should be "prediction_BioCreativeVIIITask3.tsv"'
            log.fatal(msg)
            raise Exception(msg)
    elif len(sys.argv)==1: #local call
        log.basicConfig(level=log.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        handlers=[log.StreamHandler(sys.stdout),
                                  log.FileHandler("/Users/weissenbacd/Documents/tmp/HPOEval.log", 'w', 'utf-8')])
        log.warning("Start scoring BioCreative  VIII Task 3...")
        print("Start scoring BioCreative  VIII Task 3...")
        PATH_TRUTH = '/Users/weissenbacd/tal/Ecrit/Workshops/Accepted/BioCreativeVIII23/Data/TEST_SET/TEST_SET_WITH_SOLUTION/BioCreativeVIII3_TestSet.tsv'
        output_dir = '/Users/weissenbacd/Documents/tmp/'
        subtask = "3a"
        PATH_PREDICTIONS = '/Users/weissenbacd/Documents/tmp/prediction_BioCreativeVIIITask3a.tsv'
    else:
        msg = f"I did not receive the expected number of parameters from argv, check setting: {len(sys.argv)}"
        log.fatal(msg)
        raise Exception(msg)

    #I have all parameters set, I start checking the predictions submitted and evaluating them
    eval = HPOEvaluator()
    findings = eval.evaluate(PATH_TRUTH, PATH_PREDICTIONS)
    scores = findings.getF1Scores()
    if subtask=="3a" and (("exactExtNorm_F1" in scores) or ("overExtNorm_F1" in scores)):
        msg = f"A submission with the spans predicted has been submitted to the subtask 3a, please submit the run to subtask 3b."
        log.fatal(msg)
        print(msg)
        raise Exception(msg)
    if subtask=="3b" and (("exactExtNorm_F1" not in scores) and ("overExtNorm_F1" not in scores)):
        msg = f"A submission to subtask 3b did not include the spans predictions, please submit the run to subtask 3a."
        log.fatal(msg)
        print(msg)
        raise Exception(msg)
    log.warning("scores computed.")
    print("scores computed.")

    # the scores for the leaderboard must be in a file named "scores.txt"
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        if subtask=="3a":
            output_file.write("3aNormOnlyF1: " + str(scores["Normalization_F1"]) + "\n")
            output_file.write("3aNormOnlyP: " + str(scores["Normalization_Precision"]) + "\n")
            output_file.write("3aNormOnlyR: " + str(scores["Normalization_Recall"]) + "\n")
            # output_file.write("3bexactExtNormF1: - \n")
            # output_file.write("3bexactExtNormP:  - \n")
            # output_file.write("3bexactExtNormR:  - \n")
            # output_file.write("3boverExtNormF1:  - \n")
            # output_file.write("3boverExtNormP:   - \n")
            # output_file.write("3boverExtNormR:   - \n")
            output_file.flush()
            log.warning("Output file written Task 3a. Exit.")
            print("Output file written Task 3a. Exit.")
        else:
            output_file.write("3bNormOnlyF1: " + str(scores["Normalization_F1"]) + "\n")
            output_file.write("3bNormOnlyP: " + str(scores["Normalization_Precision"]) + "\n")
            output_file.write("3bNormOnlyR: " + str(scores["Normalization_Recall"]) + "\n")
            output_file.write("3bexactExtNormF1: " + str(scores["exactExtNorm_F1"]) + "\n")
            output_file.write("3bexactExtNormP: " + str(scores["exactExtNorm_Precision"]) + "\n")
            output_file.write("3bexactExtNormR: " + str(scores["exactExtNorm_Recall"]) + "\n")
            output_file.write("3boverExtNormF1: " + str(scores["overExtNorm_F1"]) + "\n")
            output_file.write("3boverExtNormP: " + str(scores["overExtNorm_Precision"]) + "\n")
            output_file.write("3boverExtNormR: " + str(scores["overExtNorm_Recall"]) + "\n")
            output_file.flush()
            log.warning("Output file written Task 3b. Exit.")
            print("Output file written Task 3b. Exit.")


if __name__ == '__main__':
    # testscript() #Uncomment if you want to test the script
    evaluate() #comment if do not want to run the script locally
    #evaluateBioCreative() # Uncomment if you want to run the evaluation script through the Codalab interface