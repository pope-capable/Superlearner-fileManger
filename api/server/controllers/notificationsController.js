const SuperlearnerServices = require("../services/superlearner");
const NotificationServices = require("../services/notificationservice");
const { spawn } = require("child_process");
const path = require("path");
const Util = require("../utils/Utils");

const util = new Util();

class NotificationController {

  // // get noiications
  static async getNotifications(req, res) {
    if (!req.params.projectId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }
    try {
      // create the seed folders
      NotificationServices.getunseen(req.params)
        .then((superAD) => {
          util.setSuccess(200, "Unseen notifications", superAD);
          return util.send(res);
        })
        .catch((err) => {
          util.setError(400, "Error fetching notifications");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }


   // // get noiications with userId
   static async getUserUnseen(req, res) {
    if (!req.params.userId) {
      util.setError(400, "Please provide Valid details");
      return util.send(res);
    }
    try {
      // create the seed folders
      NotificationServices.getUserUnseen(req.params)
        .then((superAD) => {
          util.setSuccess(200, "Unseen notifications", superAD);
          return util.send(res);
        })
        .catch((err) => {
          util.setError(400, "Error fetching notifications");
          return util.send(res);
        });
    } catch (error) {
      util.setError(400, "An error occurred, please try again");
      return util.send(res);
    }
  }

}

module.exports = NotificationController;
