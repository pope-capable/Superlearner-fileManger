const ModelServices = require('../services/modelServices')
const Util = require('../utils/Utils');

const util = new Util();

class ModelControllers {

    // get upload folder for project
    static async getModels(req, res) {
        if (!req.params.projectId) {
            util.setError(400, 'Please provide Valid details');
            return util.send(res);
        }

        try {
            // create the seed folders
            ModelServices.getModels(req.params).then(models => {
                    util.setSuccess(200, "Models created", models);
                    return util.send(res);
            }).catch(err => {
                util.setError(400, 'Error getting models');
                return util.send(res);
            })
        } catch (error) {
            util.setError(400, 'An error occurred, please try again');
            return util.send(res);
        }
    }

}

module.exports = ModelControllers;