const express = require('express');
const router = express.Router();
const entry = require('../utils/security');
const ModelControllers = require('../controllers/modelController');

router.get('/get-all/:projectId', entry,  ModelControllers.getModels);


module.exports = router;