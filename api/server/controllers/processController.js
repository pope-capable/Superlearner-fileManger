const ProcessServices = require("../services/processes");
const FileServices = require("../services/fileServices");
const SuperlearnerServices = require("../services/superlearner");
const FolderServices = require("../services/folderServices");
const ModelServices = require("../services/modelServices");
const NotificationServices = require("../services/notificationservice");
const Util = require("../utils/Utils");
const path = require("path");
const { spawn } = require("child_process");

const util = new Util();

class ProcessController {
  // create one user folder
  static async createProcess(req, res) {
    if (!req.body.projectId || !req.body.location) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      function runMDScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/missing_DATA.py"),
          req.body.output,
          req.body.location,
          req.body.misingDataPercentage,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runFOScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/outlier.py"),
          req.body.output,
          req.body.location,
          req.body.sd,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runSFRFAScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Feature_Selection_RFE.py"),
          req.body.output,
          req.body.location,
          req.body.act_column,
          req.body.outc,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runFSBURATAcript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Feature_selection_Boruta1.py"),
          req.body.output,
          req.body.location,
          req.body.act_column,
          req.body.outc,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function linearSVMCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Preschool_Model_LinearSVM1.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function KNNCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Preschool_Model_KNN1.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function DecisionTreeCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(
            __dirname,
            "./scripts/Model Construction_ Decision Tree_1.py"
          ),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function NaiveBayesCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(
            __dirname,
            "./scripts/Model Construction_ Naive Bayes_1.py"
          ),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function MLPCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Preschool_Model_LInearSVM1.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function RFCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Preschool_Model_LInearSVM1.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function RBFSVMCreationScript(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Preschool_Model_RBFSVM.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      // create process here
      var processInf = {
        name: req.body.output,
        projectId: req.body.projectId,
        type: req.body.type,
      };
      ProcessServices.createProcess(processInf).then((process) => {
        if (req.body.value == 2) {
          var subprocess = runFOScript(process.id);
        } else if (req.body.value == 3) {
          if (req.body.fsValue == 1) {
            var subprocess = runSFRFAScript(process.id);
          } else if (req.body.fsValue == 2) {
            var subprocess = runFSBURATAcript(process.id);
          }
        } else if (req.body.value == 1) {
          var subprocess = runMDScript(process.id);
        } else if (req.body.value == 11) {
          var subprocess = linearSVMCreationScript(process.id);
        } else if (req.body.value == 12) {
          var subprocess = KNNCreationScript(process.id);
        } else if (req.body.value == 13) {
          var subprocess = DecisionTreeCreationScript(process.id);
        } else if (req.body.value == 14) {
          var subprocess = NaiveBayesCreationScript(process.id);
        } else if (req.body.value == 15) {
          var subprocess = MLPCreationScript(process.id);
        } else if (req.body.value == 16) {
          var subprocess = RFCreationScript(process.id);
        } else if (req.body.value == 17) {
          var subprocess = RBFSVMCreationScript(process.id);
        } else if (req.body.value == 1) {
          var subprocess = runMDScript(process.id);
        }

        subprocess.stdout.on("data", (data) => {
          console.log(`data:${data}`);
        });
        subprocess.stderr.on("data", (data) => {
          console.log(`error:${data}`);
        });
        subprocess.stderr.on("close", () => {
          console.log("Closed");
        });
        console.log("SCRIPT STARTED");
        util.setSuccess(200, "Process created successfully", process);
        return util.send(res);
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // create model prediction proccess
  static async createPredictionProcess(req, res) {
    if (!req.body.projectId || !req.body.location) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      function runPredictionSVMType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/LinearSVM_predict.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runPredictionKNNType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/Knn_predict1.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runPredictionDECISIONTREEType(processIdentifier) {
        console.log(
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier
        );
        return spawn("python", [
          path.join(__dirname, "./scripts/Decision_Tree_predicition_1.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runPredictionMIXEDNBType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/MixedNB_predict_1.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runPredictionRBFSVMType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/RBFSVM_predict.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      function runPredictionMLPType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/LinearSVM_predict.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }

      function runPredictionRFType(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/LinearSVM_predict.py"),
          req.body.output,
          req.body.location,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
        ]);
      }
      // create process here
      var processInf = {
        name: req.body.output,
        projectId: req.body.projectId,
        type: req.body.type,
      };
      ProcessServices.createProcess(processInf).then((process) => {
        if (req.body.modelType == "LinearSVM_Model") {
          var subprocess = runPredictionSVMType(process.id);
        } else if (req.body.modelType == "knn_Model") {
          var subprocess = runPredictionKNNType(process.id);
        } else if (req.body.modelType == "Decision_Tree") {
          var subprocess = runPredictionDECISIONTREEType(process.id);
        } else if (req.body.modelType == "Mixed_Naive_Bayes") {
          var subprocess = runPredictionMIXEDNBType(process.id);
        } else if (req.body.modelType == "RBFSVM") {
          var subprocess = runPredictionRBFSVMType(process.id);
        } else if (req.body.modelType == "Random_Forest") {
          var subprocess = runPredictionRFType(process.id);
        } else if (req.body.modelType == "MLP") {
          var subprocess = runPredictionMLPType(process.id);
        }

        util.setSuccess(200, "Prediction started successfully", process);
        return util.send(res);
        subprocess.stdout.on("data", (data) => {
          console.log(`data:${data}`);
        });
        subprocess.stderr.on("data", (data) => {
          console.log(`error:${data}`);
        });
        subprocess.stderr.on("close", () => {
          console.log("Closed");
        });
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // create superlearber creation process
  static async createSuperlearnerProcess(req, res) {
    if (!req.body.projectId || !req.body.location) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      function runSLLogisticRegression(processIdentifier) {
        console.log(
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
          req.body.compositeModels
        );
        return spawn("python", [
          path.join(__dirname, "./scripts/superLearner_LR.py"),
          req.body.output,
          req.body.location,
          req.body.studyId,
          req.body.outcomeId,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
          req.body.compositeModels,
        ]);
      }

      function runSLExtraTree(processIdentifier) {
        return spawn("python", [
          path.join(__dirname, "./scripts/superLearner_ETC.py"),
          req.body.output,
          req.body.location,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
          req.body.studyId,
          req.body.outcomeId,
          req.body.compositeModels,
        ]);
      }

      // create process here
      var processInf = {
        name: req.body.output,
        projectId: req.body.projectId,
        type: req.body.type,
      };
      ProcessServices.createProcess(processInf).then((process) => {
        if (req.body.selectedMeta == 1) {
          var subprocess = runSLLogisticRegression(process.id);
        } else {
          var subprocess = runSLExtraTree(process.id);
        }
        util.setSuccess(
          200,
          "Superlearner creating started successfully",
          process
        );
        return util.send(res);
        subprocess.stdout.on("data", (data) => {
          console.log(`data:${data}`);
        });
        subprocess.stderr.on("data", (data) => {
          console.log(`error:${data}`);
        });
        subprocess.stderr.on("close", () => {
          console.log("Closed");
        });
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // create prediction process
  static async createComparison(req, res) {
    if (!req.body.projectId || !req.body.location) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      function compareProcess(processIdentifier) {
        console.log(
          req.body.output,
          req.body.location,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
          req.body.studyId,
          req.body.outcomeId,
          req.body.comparingModels
        );
        return spawn("python", [
          path.join(__dirname, "./scripts/comparism2.py"),
          req.body.output,
          req.body.location,
          req.body.projectId,
          req.headers.token,
          processIdentifier,
          req.body.studyId,
          req.body.outcomeId,
          req.body.comparingModels,
        ]);
      }

      // create process here
      var processInf = {
        name: req.body.output,
        projectId: req.body.projectId,
        type: req.body.type,
      };
      ProcessServices.createProcess(processInf).then((process) => {
        var subprocess = compareProcess(process.id);
        util.setSuccess(200, "Comparison started successfully", process);
        return util.send(res);
        subprocess.stdout.on("data", (data) => {
          console.log(`data:${data}`);
        });
        subprocess.stderr.on("data", (data) => {
          console.log(`error:${data}`);
        });
        subprocess.stderr.on("close", () => {
          console.log("Closed");
        });
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // // get system folders
  static async getProcesses(req, res) {
    if (!req.params.projectId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      ProcessServices.getProcesses(req.params).then((processes) => {
        util.setSuccess(200, "Process Available", processes);
        return util.send(res);
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update a process
  static async finishProcess(req, res) {
    if (!req.body.location || !req.body.projectId || !req.body.processId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      FolderServices.getProjectResultFolder(req.body)
        .then((uploadTo) => {
          var fileInfo = [
            {
              folderId: uploadTo.id,
              location: req.body.location,
              name: req.body.name,
              type: req.body.type,
            },
            {
              folderId: uploadTo.id,
              location: req.body.location_two,
              name: req.body.name_two,
              type: req.body.type_two,
            },
          ];
          FileServices.createMultipleFiles(fileInfo)
            .then((fileCreated) => {
              var processInfo = {
                processId: req.body.processId,
                status: "Completed",
                result: fileCreated.id,
              };
              var notData = {
                projectId: req.body.projectId,
                content: "New process completed",
                type: "process",
              };
              NotificationServices.createNotification(notData);
              ProcessServices.updateProcess(processInfo)
                .then((processDone) => {
                  util.setSuccess(200, "Process Completed", processDone);
                  return util.send(res);
                })
                .catch((err) => {
                  util.setError(400, "Error completing process");
                  return util.send(res);
                });
            })
            .catch((err) => {
              util.setError(400, "Error creating file");
              return util.send(res);
            });
        })
        .catch((err) => {
          util.setError(400, "Error fetching files");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update data-pp process
  static async finishDPPProcess(req, res) {
    if (!req.body.location || !req.body.projectId || !req.body.processId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      FolderServices.getProjectDPPFolder(req.body)
        .then((uploadTo) => {
          var fileInfo = [
            {
              folderId: uploadTo.id,
              location: req.body.location,
              name: req.body.name,
              type: req.body.type,
            },
            {
              folderId: uploadTo.id,
              location: req.body.location_two,
              name: req.body.name_two,
              type: req.body.type_two,
            },
          ];
          FileServices.createMultipleFiles(fileInfo)
            .then((fileCreated) => {
              var processInfo = {
                processId: req.body.processId,
                status: "Completed",
                result: fileCreated.id,
              };
              var notData = {
                projectId: req.body.projectId,
                content: "New data pre-processing completed",
                type: "process",
              };
              NotificationServices.createNotification(notData);
              ProcessServices.updateProcess(processInfo)
                .then((processDone) => {
                  util.setSuccess(200, "Process Completed", processDone);
                  return util.send(res);
                })
                .catch((err) => {
                  util.setError(400, "Error completing process");
                  return util.send(res);
                });
            })
            .catch((err) => {
              util.setError(400, "Error creating file");
              return util.send(res);
            });
        })
        .catch((err) => {
          util.setError(400, "Error fetching files");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update a process with one result file
  static async finishProcessOneFile(req, res) {
    if (!req.body.location || !req.body.projectId || !req.body.processId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      FolderServices.getProjectResultFolder(req.body)
        .then((uploadTo) => {
          var fileInfo = {
            folderId: uploadTo.id,
            location: req.body.location,
            name: req.body.name,
            type: req.body.type,
          };
          FileServices.createFile(fileInfo)
            .then((fileCreated) => {
              var processInfo = {
                processId: req.body.processId,
                status: "Completed",
                result: fileCreated.id,
              };
              var notData = {
                projectId: req.body.projectId,
                content: `Process ${req.body.name} completed`,
                type: "process",
              };
              NotificationServices.createNotification(notData);
              ProcessServices.updateProcess(processInfo)
                .then((processDone) => {
                  util.setSuccess(200, "Process Completed", processDone);
                  return util.send(res);
                })
                .catch((err) => {
                  util.setError(400, "Error completing process");
                  return util.send(res);
                });
            })
            .catch((err) => {
              util.setError(400, "Error creating file");
              return util.send(res);
            });
        })
        .catch((err) => {
          util.setError(400, "Error fetching files");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update data-pp process with one result file
  static async finishDPPProcessOneFile(req, res) {
    if (!req.body.location || !req.body.projectId || !req.body.processId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      FolderServices.getProjectDPPFolder(req.body)
        .then((uploadTo) => {
          var fileInfo = {
            folderId: uploadTo.id,
            location: req.body.location,
            name: req.body.name,
            type: req.body.type,
          };
          FileServices.createFile(fileInfo)
            .then((fileCreated) => {
              var processInfo = {
                processId: req.body.processId,
                status: "Completed",
                result: fileCreated.id,
              };
              var notData = {
                projectId: req.body.projectId,
                content: `Process ${req.body.name} completed`,
                type: "process",
              };
              NotificationServices.createNotification(notData)
                .then((noecre) => {
                  console.log("MEEK", noecre);
                })
                .catch((err) => {
                  console.log("MEEK", err);
                });
              ProcessServices.updateProcess(processInfo)
                .then((processDone) => {
                  util.setSuccess(200, "Process Completed", processDone);
                  return util.send(res);
                })
                .catch((err) => {
                  util.setError(400, "Error completing process");
                  return util.send(res);
                });
            })
            .catch((err) => {
              util.setError(400, "Error creating file");
              return util.send(res);
            });
        })
        .catch((err) => {
          util.setError(400, "Error fetching files");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update a process
  static async updateFailedProcess(req, res) {
    if (!req.body.processId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }
    try {
      // re-format failed update data
      var processInfo = {
        processId: req.body.processId,
        status: "Failed",
        failure_reason: req.body.reason,
      };
      var notData = {
        projectId: req.body.projectId,
        content: `Process ${req.body.name} failed bacause: ${req.body.reason}`,
        type: "process",
      };
      NotificationServices.createNotification(notData)
        .then((noecre) => {
          console.log("MEEK", noecre);
        })
        .catch((err) => {
          console.log("MEEK", err);
        });
      ProcessServices.updateProcess(processInfo)
        .then((processDone) => {
          util.setSuccess(200, "Process Updated", processDone);
          return util.send(res);
        })
        .catch((err) => {
          util.setError(400, "Error completing process");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // update a process and create model
  static async finishModelProcessOneFile(req, res) {
    if (!req.body.location || !req.body.projectId || !req.body.processId) {
      util.setError(400, "Please provide Valid detailss");
      return util.send(res);
    }

    try {
      FolderServices.getProjectResultFolder(req.body)
        .then((uploadTo) => {
          var fileInfo = [
            {
              folderId: uploadTo.id,
              location: req.body.locationTwo,
              name: req.body.nameTwo,
              type: req.body.typeTwo,
            },
            {
              folderId: uploadTo.id,
              location: req.body.locationThree,
              name: req.body.nameThree,
              type: req.body.typeThree,
            },
            {
              folderId: uploadTo.id,
              location: req.body.locationFour,
              name: req.body.nameFour,
              type: req.body.typeFour,
            },
          ];
          FileServices.createMultipleFiles(fileInfo)
            .then((fileCreated) => {
              var modelInfo = {
                projectId: req.body.projectId,
                name: req.body.modelName,
                type: req.body.modelType,
                location: req.body.location,
                superLocation: req.body.locationFive,
              };
              ModelServices.createModel(modelInfo)
                .then((modelCreated) => {
                  var processInfo = {
                    processId: req.body.processId,
                    status: "Completed",
                    result: fileCreated[0].id,
                  };
                  var notData = {
                    projectId: req.body.projectId,
                    content: `Model ${req.body.modelName} completed`,
                    type: "process",
                  };
                  NotificationServices.createNotification(notData);
                  ProcessServices.updateProcess(processInfo)
                    .then((processDone) => {
                      util.setSuccess(200, "Process Completed", processDone);
                      return util.send(res);
                    })
                    .catch((err) => {
                      util.setError(400, "Error completing process");
                      return util.send(res);
                    });
                })
                .catch((err) => {
                  util.setError(400, "Error creating file");
                  return util.send(res);
                });
            })
            .catch((err) => {
              util.setError(400, "Error creating file");
              return util.send(res);
            });
        })
        .catch((err) => {
          util.setError(400, "Error fetching files");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }
}

module.exports = ProcessController;
