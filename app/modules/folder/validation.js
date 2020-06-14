import Joi from '@hapi/joi';

export default {
  create: {
    body : {
      schema : Joi.object({
        name: Joi.string()
          .max(250)
          .trim()
          .required()
      }),
    }
  },

  view: {
    params : {
      schema : Joi.object({
        id: Joi.number().integer().required()
      })
    }
  },

  list: {
    params : {
      schema : Joi.object({
        pageId: Joi.number().integer().default(1),
        limit : Joi.number().integer().required()
      }),
      options : {
        convert : true
      }
    }
  },

  update: {
    body : {
      schema : Joi.object({
        name: Joi.string()
          .trim()
          .required()
      }),
    },
    params : {
      schema : Joi.object({
        id : Joi.number().integer().required()
      })
    }
  },

  remove: {
    params : {
      schema : Joi.object({
        id : Joi.number().integer().required()
      })
    }
  },

};
