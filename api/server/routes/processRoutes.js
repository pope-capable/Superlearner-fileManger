const express = require("express");
const router = express.Router();
const entry = require("../utils/security");
const ProcessController = require("../controllers/processController");

router.post("/create", entry, ProcessController.createProcess);
router.post("/super-create", entry, ProcessController.createSuperlearnerProcess);
router.post("/create_prediction", entry, ProcessController.createPredictionProcess);
router.post("/compare-model", entry, ProcessController.createComparison);
router.get("/all/:projectId", entry, ProcessController.getProcesses);
router.post("/complete", entry, ProcessController.finishProcess);
router.post("/complete_dpp", entry, ProcessController.finishDPPProcess);
router.post("/complete_model", entry, ProcessController.finishModelProcessOneFile);
router.post("/complete_wone", entry, ProcessController.finishProcessOneFile);
router.post("/complete_wone_dpp", entry, ProcessController.finishDPPProcessOneFile);
router.post("/failed", entry, ProcessController.updateFailedProcess);

module.exports = router;
