const express = require('express');
const router = express.Router();
const entry = require('../utils/security');
const SuperControllers = require('../controllers/superLearnerController');

router.post('/create', entry,  SuperControllers.createOneSyperlearner);
router.get('/created/:projectId', entry,  SuperControllers.getSuperlearners);
router.post('/create_prediction', entry, SuperControllers.createPredictionProcess)


module.exports = router;