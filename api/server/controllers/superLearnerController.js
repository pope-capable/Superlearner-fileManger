const SuperlearnerServices = require("../services/superlearner");
const ProcessServices = require("../services/processes");
const FolderServices = require("../services/folderServices");
const FileServices = require("../services/fileServices");
const NotificationServices = require("../services/notificationservice")
const { spawn } = require("child_process");
const path = require("path");
const Util = require("../utils/Utils");

const util = new Util();

class SupperLearnerController {
  // create one superlearner
  static async createOneSyperlearner(req, res) {
    if (!req.body.projectId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      // create superlearner here
      SuperlearnerServices.createSuperlearner(req.body).then((newFile) => {
          FolderServices.getProjectResultFolder(req.body).then((uploadTo) => {
              var fileInfo = {
                  folderId: uploadTo.id,
                  location: req.body.location3,
                  name: req.body.name3,
                  type: req.body.type3,
                }
              FileServices.createFile(fileInfo)
                .then((fileCreated) => {
                  var processInfo = {
                    processId: req.body.processId,
                    status: "Completed",
                  };
                  var notData = {
                    projectId: req.body.projectId, content: `Super learner ${req.body.name3} created`, type: "process"
                  }
                  NotificationServices.createNotification(notData)
                  ProcessServices.updateProcess(processInfo)
                    .then((processDone) => {
                      util.setSuccess(
                        200,
                        "Superlearner created Successfully",
                        processDone
                      );
                      return util.send(res);
                    })
                    .catch((err) => {
                      util.setError(400, "Error completing process");
                      return util.send(res);
                    });
                }).catch((err) => {
                  util.setError(400, "Cannot create file");
                  return util.send(res);
                });
            }).catch((err) => {
              util.setError(400, "Cannot find project result folder");
              return util.send(res);
            });
        })
      .catch((err) => {
        util.setError(400, "Error creating superlearner");
        return util.send(res);
      });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // // get system folders
  static async getSuperlearners(req, res) {
    if (!req.params.projectId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      // create the seed folders
      SuperlearnerServices.getSuperLearners(req.params)
        .then((superAD) => {
          util.setSuccess(200, "Superlearners", superAD);
          return util.send(res);
        })
        .catch((err) => {
          util.setError(400, "Error fetching superlearners");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

  // create model prediction proccess
  static async createPredictionProcess(req, res) {
    if (!req.body.projectId || !req.body.uploadedFile) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }

    try {
      function PredictSL(processIdentifier) {
        console.log(
          req.body.output,
          req.body.location1,
          req.body.location2,
          req.body.uploadedFile,
          req.body.projectId,
          req.headers.token,
          processIdentifier
        );
        return spawn("python", [
          path.join(__dirname, "./scripts/SuperLearner_Predict.py"),
          req.body.output,
          req.body.location1,
          req.body.location2,
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
      ProcessServices.createProcess(processInf)
        .then((process) => {
          var subprocess = PredictSL(process.id);
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
        })
        .catch((err) => {
          util.setError(400, "Process creation failed");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }
}

module.exports = SupperLearnerController;
