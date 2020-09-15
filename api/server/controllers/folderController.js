const FolderServices = require('../services/folderServices')
const Util = require('../utils/Utils');

const util = new Util();

class FolderControllers {

    // seed system folders
    static async seedSystemFolders(req, res) {
        if (!req.body.projectId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            var projectIdentifier = req.body.projectId
            var systemFolders = [{name: "Uploads", type: "system", projectId: projectIdentifier}, {name: "Pre-processed files", type: "system", projectId: projectIdentifier}, {name: "Results", type: "system", projectId: projectIdentifier}]
            FolderServices.createSystemFolders(systemFolders).then(seededFolders => {
                    util.setSuccess(200, "Project folders created", seededFolders);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error creating folders');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

    // create one user folder
    static async createNewFolder(req, res) {
        if (!req.body.projectId || !req.body.name) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            FolderServices.createFolder(req.body).then(seededFolders => {
                    util.setSuccess(200, "New folder created", seededFolders);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error creating folders');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

    // get system folders
    static async getProjectFolders(req, res) {
        if (!req.params.projectId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            FolderServices.getProjectFolders(req.params).then(folders => {
                    util.setSuccess(200, "ProjectFolders available", folders);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error creating folders');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

    // get upload folder for project
    static async getProjecUF(req, res) {
        if (!req.params.projectId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            FolderServices.getProjectUploadFolder(req.params).then(folders => {
                    util.setSuccess(200, "ProjectFolders available", folders);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error creating folders');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

}

module.exports = FolderControllers;