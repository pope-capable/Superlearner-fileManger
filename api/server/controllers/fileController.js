const FileServices = require('../services/fileServices')
const Util = require('../utils/Utils');

const util = new Util();

class FolderControllers {

    // create one user folder
    static async uploadOneFile(req, res) {
        if (!req.body.folderId || !req.file) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create file here
            var useFile = {folderId: req.body.folderId, name: req.file.originalname, type: req.file.originalname.split('.').pop(), location: req.file.location}
            FileServices.createFile(useFile).then(newFile => {
                    util.setSuccess(200, "File Uploaded Successfully", newFile);
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

    // // get system folders
    static async getFolderFiles(req, res) {
        if (!req.params.folderId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            FileServices.getFiles(req.params).then(filesAD => {
                    util.setSuccess(200, "Files", filesAD);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error fetching files');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

    // get all recent files
    static async getRecentFolderFiles(req, res) {
        if (!req.params.projectId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            FileServices.getRecentFiles(req.params).then(filesAD => {
                    util.setSuccess(200, "Files", filesAD);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error fetching files');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

}

module.exports = FolderControllers;